import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from models.KDNet_model_auto import KDNet, ViTeacher, AuTeacher
from models.my_dataset0726 import myDataset
from tools import ops
import argparse

#
# '''
# 更新日志：
# 拷贝自0717
# 增加单独训练教师的功能
# '''

log_id = 'log_auto'
if os.path.exists(log_id) is False:
    os.makedirs(log_id)


class KDNetTrainer():
    def main(self):
        self.loaddata()
        self.setmodel()
        self.train()

    def __init__(self, gpu, fa_a, fa_v, fg_a, fg_v, lg_a, lg_v, train_vi_teacher=False, train_au_teacher=False, decay=False, attention_name='CMHA'):
        self.fa_a = fa_a
        self.fa_v = fa_v
        self.fg_a = fg_a
        self.fg_v = fg_v
        self.lg_a = lg_a
        self.lg_v = lg_v
        self.decay = decay
        self.attention_name = attention_name

        self.distill_rate = 0.0 if train_vi_teacher or train_au_teacher else 0.3
        print(self.distill_rate)
        self.train_vi_teacher = train_vi_teacher
        self.train_au_teacher = train_au_teacher
        self.device_ids = [gpu]
        '''set seq and path'''
        self.trainSeqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000']
        self.datasetPath = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/STNet'
        '''set log path'''
        self.date = 'kdnet'
        self.BASE_DIR = './'
        self.log_dir = os.path.abspath(os.path.join(self.BASE_DIR, log_id, "_{0}_model_{1}.pth".format(self.date, self.attention_name)))
        # self.checkpoint_path = os.path.abspath(os.path.join(self.BASE_DIR, 'log6', 'checkpoint.pth'))
        if train_au_teacher:
            ckpt_path = 'au_ckpt.pth'
        elif train_vi_teacher:
            ckpt_path = 'vi_ckpt.pth'
        else:
            ckpt_path = f'{fa_a}_{fa_v}_{fg_a}_{fg_v}_{lg_a}_{lg_v}{"_decay" if self.decay else ""}_ckpt.pth'
        # self.checkpoint_path = os.path.abspath(os.path.join(self.BASE_DIR, log_id, 'model_checkpoint.pth'))
        self.checkpoint_path = os.path.abspath(os.path.join(self.BASE_DIR, log_id, ckpt_path))
        '''set flag : train/test'''
        self.train_flag = True
        self.saveNetwork_flag = True
        self.drawCurve_flag = True
        ops.set_seed(1)
        self.MAX_EPOCH = 1
        self.BATCH_SIZE = 8
        self.LR = 0.0001
        self.log_interval = 16
        self.val_interval = 1
        self.save_interval = 1

    def loaddata(self):
        trainList, validList = ops.splitDataset(self.datasetPath, self.trainSeqList, splitType='train&valid',
                                                trainPct=0.8)
        print(len(trainList), len(validList))
        refpath = f'{self.datasetPath}/AVsample/ref_seq123.pkl'
        with open(refpath, 'rb') as data:
            refData = pickle.load(data)
        train_data = myDataset(dataList=trainList, refData=refData)
        valid_data = myDataset(dataList=validList, refData=refData)
        self.train_loader = DataLoader(dataset=train_data, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
        self.valid_loader = DataLoader(dataset=valid_data, batch_size=self.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    def setmodel(self):
        net_path_vi = os.path.join(self.BASE_DIR, 'log0731/vi_kdnet_ep3.pth')
        net_path_au = os.path.join(self.BASE_DIR, 'log0731/au_kdnet_ep3.pth')
        # net = KDNet(net_path_vi=net_path_vi, net_path_au=net_path_au)
        if self.train_au_teacher:
            net = AuTeacher()
        elif self.train_vi_teacher:
            net = ViTeacher()
        else:
            net = KDNet(self.fa_a, self.fa_v, self.fg_a, self.fg_v, self.lg_a, self.lg_v, net_path_vi=net_path_vi, net_path_au=net_path_au, decay=self.decay, attention_name=self.attention_name)
        net = torch.nn.DataParallel(net, device_ids=self.device_ids)
        self.net = net.cuda(device=self.device_ids[0])
        torch.enable_grad()
        # self.lossFn1 = nn.MSELoss(reduction='mean')
        self.lossFn1 = nn.MSELoss()
        delta = 1.0  # 选择适当的 delta 值
        self.huber_loss = nn.SmoothL1Loss(beta=delta)
        self.lossFn2 = nn.MSELoss(reduction='mean')
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.LR)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Load checkpoint if it exists
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=f'cuda:{self.device_ids[0]}')
            self.net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.start_epoch = 0

    def train(self):
        if self.train_flag:
            train_curve = list()
            valid_curve = list()
            from tqdm import tqdm
            for epoch in range(self.start_epoch, self.MAX_EPOCH):
                loss_mean = 0.
                dist_mean = 0.
                self.net.train()
                for i, data in tqdm(enumerate(self.train_loader)):
                    refImg, sampleImg0, sampleImg1, sampleImg2, GCFmap, sampleFace, segImg = data
                    imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    auFr = Variable(torch.as_tensor(GCFmap, dtype=torch.float32), requires_grad=True) \
                        .cuda(device=self.device_ids[0])
                    labels = Variable(torch.as_tensor(sampleFace, dtype=torch.float32), requires_grad=False) \
                        .cuda(device=self.device_ids[0])
                    # 梯度置零
                    self.optimizer.zero_grad()

                    # 前向传播
                    # outputs, evl_factor, total_loss = self.net(imgRef, img0, img1, img2, auFr)
                    outputs, distill_loss = self.net(imgRef, img0, img1, img2, auFr)
                    loss1 = self.huber_loss(outputs, labels)
                    r = outputs.detach()
                    # print(labels)
                    # print(r)
                    dist = torch.sqrt(torch.sum((r - labels) ** 2, axis=1)).mean()
                    # print(dist)


                    # label2 = torch.div(2, torch.exp(0.05 * dist) + 1)
                    # loss2 = self.lossFn2(evl_factor.squeeze(), label2)
                    total_loss = torch.mean((1.0 - self.distill_rate) * loss1 + self.distill_rate * distill_loss)

                    # 反向传播和参数更新
                    total_loss.backward()
                    self.optimizer.step()

                    # 记录损失
                    loss_mean += total_loss.item()
                    dist_mean += torch.mean(dist).item()
                    train_curve.append(total_loss.item())
                    if (i + 1) % self.log_interval == 0:
                        loss_mean = loss_mean / self.log_interval
                        dist_mean = dist_mean / self.log_interval
                        print("[{}] Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                            self.date, epoch, self.MAX_EPOCH, i + 1, len(self.train_loader), loss_mean))
                        print("MAE: {:.4f}".format(dist_mean))
                        loss_mean = 0.
                        dist_mean = 0.
                # 调整学习率
                self.scheduler.step()

                # 验证模型
                if (epoch + 1) % self.val_interval == 0:
                    loss_val = 0.
                    dist_val = 0.
                    with torch.no_grad():
                        for j, data in enumerate(self.valid_loader):
                            refImg, sampleImg0, sampleImg1, sampleImg2, GCFmap, sampleFace, segImg = data
                            imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            auFr = Variable(torch.as_tensor(GCFmap, dtype=torch.float32), requires_grad=True).cuda(
                                device=self.device_ids[0])
                            labels = Variable(torch.as_tensor(sampleFace, dtype=torch.float32), requires_grad=False) \
                                .cuda(device=self.device_ids[0])
                            # 梯度置零
                            self.optimizer.zero_grad()

                            # 前向传播
                            # outputs, evl_factor, total_loss = self.net(imgRef, img0, img1, img2, auFr)
                            outputs, distill_loss = self.net(imgRef, img0, img1, img2, auFr)
                            loss1 = self.huber_loss(outputs, labels)
                            r = outputs.detach()
                            # print(labels)

                            dist = torch.sqrt(torch.sum((r - labels) ** 2, axis=1))
                            # label2 = torch.div(2, torch.exp(0.05 * dist) + 1)
                            # loss2 = self.lossFn2(evl_factor.squeeze(), label2)
                            # total_loss = torch.mean(total_loss + loss1 + loss2)
                            total_loss = torch.mean((1.0 - self.distill_rate) * loss1 + self.distill_rate * distill_loss)

                            # 记录损失
                            loss_val += total_loss.item()
                            dist_val += torch.mean(dist).item()


                    # 计算并记录验证集损失
                    loss_val_epoch = loss_val / len(self.valid_loader)
                    dist_val_epoch = dist_val / len(self.valid_loader)
                    valid_curve.append(loss_val_epoch)
                    print("Valid: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                        epoch, self.MAX_EPOCH, j + 1, len(self.valid_loader), loss_val_epoch))
                    print("MAE: {:.4f}".format(dist_val_epoch))

                # 保存模型
                if (epoch + 1) % self.save_interval == 0:
                    if self.train_vi_teacher:
                        model = 'vi'
                    elif self.train_au_teacher:
                        model = 'au'
                    else:
                        # model = f'{fa_a}_{fa_v}_{fg_a}_{fg_v}_{lg_a}_{lg_v}_model'
                        model = f'{fa_a}_{fa_v}_{fg_a}_{fg_v}_{lg_a}_{lg_v}{"_decay" if self.decay else ""}_{self.attention_name}_model'

                    log_dir = os.path.abspath(
                        os.path.join(self.BASE_DIR, log_id, f"{model}_{self.date}_ep{epoch}.pth"))
                    state = {'model': self.net.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
                    torch.save(state, log_dir)
                    # torch.save(state, self.checkpoint_path)

            # 绘制损失曲线
            if self.drawCurve_flag:
                train_x = range(len(train_curve))
                train_y = train_curve
                train_iters = len(self.train_loader)
                valid_x = [i * train_iters * self.val_interval for i in range(1, len(valid_curve) + 1)]
                valid_y = valid_curve
                plt.plot(train_x, train_y, label='Train')
                plt.plot(valid_x, valid_y, label='Valid')
                plt.legend(loc='upper right')
                plt.ylabel('Loss value')
                plt.xlabel('Iteration')
                plt.show()
                print('End training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--decay', type=bool, default=False, help='lg_decay')
    parser.add_argument('--attention_name', type=str, default='CMHA', help='attention name')
    parser.add_argument('--fa_a', type=float, help='')
    parser.add_argument('--fa_v', type=float, help='')
    parser.add_argument('--fg_a', type=float, help='')
    parser.add_argument('--fg_v', type=float, help='')
    parser.add_argument('--lg_a', type=float, help='')
    parser.add_argument('--lg_v', type=float, help='')
    # 解析命令行参数
    args = parser.parse_args()
    print(args)
    gpu, decay, attention_name, fa_a, fa_v, fg_a, fg_v, lg_a, lg_v = args.gpu, args.decay, args.attention_name, args.fa_a, args.fa_v, args.fg_a, args.fg_v, args.lg_a, args.lg_v
    kdnet_trainer = KDNetTrainer(gpu, fa_a, fa_v, fg_a, fg_v, lg_a, lg_v, train_au_teacher=False, decay=decay, attention_name=attention_name)
    kdnet_trainer.main()