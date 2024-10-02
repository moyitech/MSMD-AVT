from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
import numpy as np
from tools import ops
import cv2
import math
import torchvision.models as models
from torchvision.models import resnet50



T = 1.0  # 温度参数


class ViTeacher(nn.Module):
    def __init__(self, net_path_vi=None):
        super().__init__()
        self.net1 = Net(
            backbone=AlexNetV1(),
            head=SiamFC())
        ops.init_weights(self.net1)

        self.teacher_vi_predictor = Predictor()
        ops.init_weights(self.teacher_vi_predictor)

        self.LN_vi = nn.LayerNorm([1225, 256], eps=1e-6)


    def forward(self, ref, img0, img1, img2, auFr):
        Fvi_teacher, stage_1 = self.net1(ref, img0, img1, img2)
        # print(Fvi_teacher.shape)
        # 融合对齐后的学生视觉和听觉
        Fvi_teacher = Fvi_teacher.flatten(2).permute(0, 2, 1)
        # print(Fvi_teacher.shape)
        stage_2 = self.LN_vi(Fvi_teacher)
        Fvi_teacher_output = self.teacher_vi_predictor(stage_2).squeeze(1)
        # print(Fvi_teacher_output.shape)

        return Fvi_teacher_output, stage_2, stage_1


class AuTeacher(nn.Module):
    def __init__(self, net_path_au=None):
        super().__init__()
        self.net2 = audioNet(
            net_head=AlexNetV1_au(),
            predNet=GCFpredictor())
        ops.init_weights(self.net2)

        self.teacher_au_predictor = Predictor()
        ops.init_weights(self.teacher_au_predictor)

        self.LN_au = nn.LayerNorm([1225, 256], eps=1e-6)

    def forward(self, ref, img0, img1, img2, auFr):
        Fau_teacher, stage_1 = self.net2(auFr)
        Fau_teacher = Fau_teacher.permute(0, 3, 1, 2)
        b = ref.shape[0]
        Fau_teacher = Fau_teacher.permute(0, 2, 3, 1).view(b, -1, 256)
        stage_2 = self.LN_au(Fau_teacher)
        Fau_teacher_output = self.teacher_au_predictor(stage_2)
        Fau_teacher_output = Fau_teacher_output.squeeze(1)
        return Fau_teacher_output, stage_2, stage_1


class KDNet(nn.Module):
    def __init__(self, fa_a, fa_v, fg_a, fg_v, lg_a, lg_v, net_path_vi=None, net_path_au=None, net_path_au_student=None, decay=False, attention_name='CMHA'):
        super(KDNet, self).__init__()
        self.fa_a = fa_a
        self.fa_v = fa_v
        self.fg_a = fg_a
        self.fg_v = fg_v
        self.lg_a = lg_a
        self.lg_v = lg_v
        self.decay = decay
        self.attention_name = attention_name

        self.s3_rate = 1.0
        # 视觉教师模型 net1
        # self.net1 = Net(
        #     backbone=AlexNetV1(),
        #     head=SiamFC())
        # ops.init_weights(self.net1)

        self.net1 = ViTeacher()
        if net_path_vi is not None:
            # 加载预训练模型参数
            pretrained_dict = torch.load(net_path_vi, map_location=lambda storage, loc: storage)['model']

            # 创建一个新的state_dict，手动调整键名前缀
            model_dict = self.net1.state_dict()
            for k, v in pretrained_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 去掉'module.'前缀
                if k in model_dict:
                    model_dict[k] = v
            self.net1.load_state_dict(
                model_dict
            )  # 模型是CPU，预加载的训练参数是GPU


        # 视觉学生模型 net3
        self.net3 = ModifiedResNet50()  # 使用预训练的 ResNet50 模型

        self.net2 = AuTeacher()
        if net_path_au is not None:
            pretrained_dict = torch.load(net_path_au, map_location=lambda storage, loc: storage)['model']

            # 创建一个新的state_dict，手动调整键名前缀
            model_dict = self.net2.state_dict()
            for k, v in pretrained_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # 去掉'module.'前缀
                if k in model_dict:
                    model_dict[k] = v
            self.net2.load_state_dict(
                model_dict
            )  # 模型是CPU，预加载的训练参数是GPU

        # # 教师不进行更新
        # for param in self.net1.parameters():
        #     param.requires_grad = False
        #
        # for param in self.net2.parameters():
        #     param.requires_grad = False

        # 学生听觉模型 net4
        self.net4 = audiostudentNet()
        ops.init_weights(self.net4)
        if net_path_au_student is not None:
            self.net4.load_state_dict(torch.load(net_path_au_student))

        self.netMHA = MultiHeadAttention()  # 初始化多头注意力机制模块
        ops.init_weights(self.netMHA)
        self.predNet = Predictor()
        ops.init_weights(self.predNet)
        self.evlNet = evlNet()  # 初始化评估模块
        ops.init_weights(self.evlNet)
        self.PE_vi = PositionEmbeddingSine()  # 初始化视听位置嵌入模块，用于引入位置信息
        self.PE_au = PositionEmbeddingSine()
        # 使用了 LayerNorm 层来对不同的特征进行归一化处理
        self.LN1_vi = nn.LayerNorm([256, 35, 35], eps=1e-6)
        self.LN1_au = nn.LayerNorm([256, 35, 35], eps=1e-6)  # 表示视觉和听觉特征的通道数为256，特征图的大小为35x35
        self.LN_vi = nn.LayerNorm([1225, 256], eps=1e-6)
        self.LN_au = nn.LayerNorm([1225, 256], eps=1e-6)  # 特征图的像素数量为1225，通道数为256
        self.LN_av = nn.LayerNorm([1225, 256], eps=1e-6)  # 对视觉和听觉特征的融合结果进行归一化处理
        self.LN2 = nn.LayerNorm([1225, 256], eps=1e-6)  # 对融合结果再次进行归一化处理

        self.teacher_vi_predictor = Predictor()
        ops.init_weights(self.teacher_vi_predictor)
        self.teacher_au_predictor = Predictor()
        ops.init_weights(self.teacher_au_predictor)
        if self.attention_name == 'SCMHA':
            self.attention_vi = Attention()
            self.attention_au = Attention()
        elif self.attention_name == 'NONE':
            self.attention_linear = nn.Linear(256 * 2, 256)

    def forward(self, ref, img0, img1, img2, auFr):
        b = ref.shape[0]
        c = 256

        Fvi_teacher_output, Fvi_teacher_feature, Fvi_stage_1 = self.net1(ref, img0, img1, img2, auFr)
        Fau_teacher_output, Fau_teacher_feature, Fau_stage_1 = self.net2(ref, img0, img1, img2, auFr)

        # 视觉学生模型 net3
        Fvi_student = self.net3(ref, img0, img1, img2)
        Fvi_student = self.LN1_vi(Fvi_student)

        # 听觉学生模型
        Fau_student = self.net4(auFr)
        Fau_student = self.LN1_au(Fau_student)

        Fvi_student_pe = Fvi_student + self.PE_vi(Fvi_student)
        Fau_student_pe = Fau_student + self.PE_au(Fau_student)

        Fvi_student_pe = Fvi_student_pe.permute(0, 2, 3, 1).view(b, -1, c)  # [b,N=h*w,c]
        Fau_student_pe = Fau_student_pe.permute(0, 2, 3, 1).view(b, -1, c)  # [b,N=h*w,c]

        Fvi_student2 = self.LN_vi(Fvi_student_pe)
        Fau_student2 = self.LN_au(Fau_student_pe)

        # print(Fvi_student2.shape, Fvi_student2.shape)
        # [8, 1225, 256] -> [8, 1225, 256]
        if self.attention_name == 'NONE':
            out_av = self.attention_linear(torch.cat([Fvi_student2, Fau_student2], dim=2))
        else:
            if self.attention_name == 'SCMHA':
                Fvi_student2 = self.attention_vi(Fvi_student2)
                Fau_student2 = self.attention_au(Fau_student2)

            out_vi = self.netMHA(q=Fau_student2, k=Fvi_student2, v=Fvi_student2)

            if self.attention_name == 'MHA':
                out_au = out_vi
            else:
                out_au = self.netMHA(q=Fvi_student2, k=Fau_student2, v=Fau_student2)

            out_av = out_vi + out_au



        # print(out_av.shape)
        out_av = self.LN_av(out_av)
        out_pred = self.predNet(out_av)
        out_pred = out_pred.squeeze(1)

        # stage 1
        vision_s1_loss = self.fa_v * F.mse_loss(Fvi_student, Fvi_stage_1.detach())
        audio_s1_loss = self.fa_a * F.mse_loss(Fau_student.permute(0, 2, 3, 1), Fau_stage_1.detach())

        # stage 2
        vision_feature_loss = self.fg_v * F.mse_loss(out_av, Fvi_teacher_feature.detach())
        audio_feature_loss = self.fg_a * F.mse_loss(out_av, Fau_teacher_feature.detach())

        # stage 3
        vision_logit_loss = self.lg_v * F.mse_loss(out_pred, Fvi_teacher_output.detach())
        audio_logit_loss = self.lg_a * F.mse_loss(out_pred, Fau_teacher_output.detach())
        if self.decay:
            self.s3_rate *= 0.98

        distill_loss = (1.0 * (vision_feature_loss + audio_feature_loss)
                        + self.s3_rate * (vision_logit_loss + audio_logit_loss)
                        + 1.0 * (vision_s1_loss + audio_s1_loss))

        return out_pred, distill_loss


class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的全局平均池化层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((35, 35))  # 使用自适应平均池化层，将输出大小设置为 35x35
        self.conv = nn.Conv2d(2048, 256, kernel_size=1)  # 添加一个卷积层来获得大小为 256x35x35 的特征图

    def forward(self, z, x0, x1, x2):
        z = self.features(z)
        z = self.avgpool(z)
        z = self.conv(z)

        x0 = self.features(x0)
        x0 = self.avgpool(x0)
        x0 = self.conv(x0)

        x1 = self.features(x1)
        x1 = self.avgpool(x1)
        x1 = self.conv(x1)

        x2 = self.features(x2)
        x2 = self.avgpool(x2)
        x2 = self.conv(x2)

        return x0+x1+x2


class teacherviEnocder(nn.Module):  # 在上面stnet中定义
    def __init__(self, backbone):
        super(teacherviEnocder, self).__init__()
        self.backbone = backbone

    def updownsample(self, x):
        return F.interpolate(x,size=(35,35),mode='bilinear',align_corners=False)  # 将图片的大小改为35*35

    def forward(self, z, x0, x1, x2):
        z = self.backbone(z)

        fx0 = self.backbone(x0)
        fx1 = self.backbone(x1)
        fx2 = self.backbone(x2)

        n0 = self.updownsample(fx0)
        n1 = self.updownsample(fx1)
        n2 = self.updownsample(fx2)

        return n0+n1+n2


class _AlexNet(nn.Module):  # 视觉backbones

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):  # 视觉backbones
    output_stride = 8
    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class Net(nn.Module):  # 在上面stnet中定义
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head

    def updownsample(self, x):
        return F.interpolate(x,size=(35,35),mode='bilinear',align_corners=False)  # 将图片的大小改为35*35

    def forward(self, z, x0, x1, x2):
        z = self.backbone(z)

        fx0 = self.backbone(x0)
        fx1 = self.backbone(x1)
        fx2 = self.backbone(x2)

        h0 = self.head(z, fx0)
        h1 = self.head(z, fx1)
        h2 = self.head(z, fx2)

        n0 = self.updownsample(h0)
        n1 = self.updownsample(h1)
        n2 = self.updownsample(h2)
        return n0+n1+n2, n0+n1+n2


class SiamFC(nn.Module):

    def __init__(self, out_scale=0.001):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return self._fast_xcorr(z, x) * self.out_scale

    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        z0 = z[0]
        x0 = x[0]
        out = F.conv2d(x0.unsqueeze(0), z0.unsqueeze(1), groups=c)

        for i in range(1, nz):
            zi = z[i]
            xi = x[i]
            outi = F.conv2d(xi.unsqueeze(0), zi.unsqueeze(1), groups=c)
            out = torch.cat([out, outi], dim=0)

        return out


class _AlexNet(nn.Module):  # 视觉backbones

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):  # 视觉backbones
    output_stride = 8
    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class audiostudentNet(nn.Module):
    def __init__(self):
        super(audiostudentNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 50 * 50, 512)  # 根据输入输出的维度进行调整
        self.fc2 = nn.Linear(512, 256 * 35 * 35)  # 输出维度调整为 256 x 35 x 35
        self.relu = nn.ReLU()

    def forward(self, x):  # [16, 3, 400, 400]
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # [16, 64, 200, 200]
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # [16, 128, 100, 100]
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # [16, 256, 50, 50]
        x = x.view(x.size(0), -1)  # 将特征展平成一维向量，自动计算特征维度大小
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = x.view(x.size(0), 256, 35, 35)  # 重新reshape成 [batch_size, channels, height, width] 的形状
        return x


class audioNet(nn.Module):
    def __init__(self, net_head, predNet):
        super(audioNet, self).__init__()
        self.net_head = net_head
        self.predNet = predNet

    def forward(self, x):
        stage_1 = self.net_head(x).permute(0, 2, 3, 1)  # [b,h,w,c]
        x = self.predNet(stage_1)
        return x, stage_1


class GCFpredictor(nn.Module):  # 具体network在GCFnet中介绍
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        return x


class _AlexNet_au(nn.Module):  # GCFnet中定义

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class _BatchNorm2d(nn.BatchNorm2d):  # GCFnet中定义

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)


class AlexNetV1_au(_AlexNet_au):  # GCFnet中定义
    output_stride = 8
    def __init__(self):
        super(AlexNetV1_au, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 6, 1, groups=2))


class Predictor(nn.Module):  # 将归一化处理后的特征再次输入预测
    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=(1225, 1), stride=1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            # nn.Dropout(0.1)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU(),
            # nn.Dropout(0.1)
        )
        self.fc3 = nn.Linear(256, 2, bias=False)

    def forward(self, x):
        x = self.maxPool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class Attention(nn.Module):
    # [8, 1225, 256] -> [8, 1225, 256]
    def __init__(self):
        super(Attention, self).__init__()
        self.q = nn.Linear(256, 256)
        self.k = nn.Linear(256, 256)
        self.v = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.fc(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):  # 两个参数 temperature 和 attn_dropout，分别用于指定缩放因子和注意力层中的 dropout 比例
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output


class MultiHeadAttention(nn.Module):  # 将视觉和听觉的特征融合

    def __init__(self, n_head=8, d_model=256, d_k=32, d_v=32, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = k

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):# [b,c,h,w]
        b,h,w = x.shape[0],x.shape[2],x.shape[3]
        mask = torch.ones((b, h, w), dtype=torch.bool).to(x.device)
        assert mask is not None
        not_mask = mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class evlNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=(1225, 1), stride=1)
        self.avgPool = nn.AvgPool2d(kernel_size=(1225, 1), stride=1)
        self.fc1 = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.ReLU())
        self.fc3 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x1 = self.maxPool(x)
        x2 = self.avgPool(x)
        x = x1+x2
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    # attn_model = Attention()
    # x = torch.randn(8, 1225, 256)
    # out = attn_model(x)
    # print(out.shape)
    model = ModifiedResNet50()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Vi Stu: {total_params}")

    model = audiostudentNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Au Stu: {total_params}")

    model = Net(
            backbone=AlexNetV1(),
            head=SiamFC())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Vi The: {total_params}")

    model = audioNet(
            net_head=AlexNetV1_au(),
            predNet=GCFpredictor())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Au The: {total_params}")

    # ViTeacher
