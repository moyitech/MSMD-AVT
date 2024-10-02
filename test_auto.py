import os
import glob
import random
import cv2
import numpy as np
from tools import ops as ops1
from tools import prepareTools
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import pickle
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_stGCF import getGCF
from models.KDNet_model_auto import KDNet
import argparse

parser = argparse.ArgumentParser(description="测试参数")
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--decay', type=bool, default=False, help='lg_decay')
parser.add_argument('--attention_name', type=str, default='CMHA', help='attention name')

parser.add_argument('--fa_a', type=float, help='')
parser.add_argument('--fa_v', type=float, help='')
parser.add_argument('--fg_a', type=float, help='')
parser.add_argument('--fg_v', type=float, help='')
parser.add_argument('--lg_a', type=float, help='')
parser.add_argument('--lg_v', type=float, help='')
args = parser.parse_args()
print(args)
gpu, decay, attention_name, fa_a, fa_v, fg_a, fg_v, lg_a, lg_v = args.gpu, args.decay, args.attention_name, args.fa_a, args.fa_v, args.fg_a, args.fg_v, args.lg_a, args.lg_v


def getdata(ref_examplar, sample, GCFmap):
    minGCF = -0.01
    maxGCF = 0.50
    refImg = ref_examplar.transpose(2, 0, 1)  # output is a ndarray[W*H*C]
    sampleImg0 = cv2.resize(sample, (300, 300)
                            , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    sampleImg1 = cv2.resize(sample, (400, 400)
                            , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
    sampleImg2 = cv2.resize(sample, (550, 550)
                            , interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    GCF_nor = (GCFmap - minGCF) / (maxGCF - minGCF)
    GCF_nor_expanded = np.expand_dims(GCF_nor, axis=-1)
    GCF_nor = np.repeat(GCF_nor_expanded, 3, axis=-1)
    reGCFmap = cv2.resize(GCF_nor, (400, 400), interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)

    imgRef = Variable(torch.as_tensor(refImg, dtype=torch.float32).unsqueeze(0), requires_grad=False) \
        .cuda(device=device_ids[0])
    img0 = Variable(torch.as_tensor(sampleImg0, dtype=torch.float32).unsqueeze(0), requires_grad=True) \
        .cuda(device=device_ids[0])
    img1 = Variable(torch.as_tensor(sampleImg1, dtype=torch.float32).unsqueeze(0), requires_grad=True) \
        .cuda(device=device_ids[0])
    img2 = Variable(torch.as_tensor(sampleImg2, dtype=torch.float32).unsqueeze(0), requires_grad=True) \
        .cuda(device=device_ids[0])
    auFr = Variable(torch.as_tensor(reGCFmap, dtype=torch.float32).unsqueeze(0), requires_grad=False) \
        .cuda(device=device_ids[0])
    return imgRef, img0, img1, img2, auFr

date = 'stnet'
device_ids = [gpu]
random_seed = 1
random.seed(random_seed)
seqList = ['seq08-1p-0100', 'seq11-1p-0100','seq12-1p-0100'] #'seq08-1p-0100', 'seq11-1p-0100','seq12-1p-0100'
datasetPath_vi = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/AV163'
datasetPath = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/STNet'
s_size = 120
au_observe = getGCF()
'''set log path'''
BASE_DIR = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/'
# log_dir = os.path.abspath(os.path.join(BASE_DIR, 'log0731', 'model_kdnet_ep0.pth'))
log_dir = os.path.abspath(os.path.join(BASE_DIR, 'log_auto', f'{fa_a}_{fa_v}_{fg_a}_{fg_v}_{lg_a}_{lg_v}{"_decay" if decay else ""}_{attention_name}_model_kdnet_ep0.pth'))
# ============================ load network============================
# net = KDNet(net_path_vi = None, net_path_au=None)
# net = ViTeacher()
# net = AuTeacher()
net = KDNet(fa_a, fa_v, fg_a, fg_v, lg_a, lg_v, attention_name=attention_name)
net = torch.nn.DataParallel(net, device_ids=device_ids)
net = net.cuda(device=device_ids[0])
checkpoint = torch.load(log_dir, map_location=torch.device(f'cuda:{device_ids[0]}'))
net.load_state_dict(checkpoint['model'])
net.eval()
results = dict()
#============================ tracking =========================================
total_result = list()
for sequence in seqList:
    results[sequence] = {}
    # load audio data
    audioDATA = ops1.loadaudioDATA(sequence, datasetPath)
    GCCdata = ops1.loadGCC(sequence, datasetPath)
    for cam_number in range(1, 4):#(1, 4)1-index
        results[sequence][f'cam{cam_number}'] = {}
        cnt = 0
        hit = 0

        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']

        startfr, endfr = ops1.getSE(sequence, cam_number)#1-index
        seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        total_file = sorted(glob.glob(seq_dir + '*.jpg'))
        total_GT = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                              f'{sequence}_cam{cam_number}_GT2D.txt')
        # 1.ref(path) is from the previous frame of the startfr
        ref_file = total_file[startfr - 2]
        # 2.ref_anno
        ref_anno = total_GT[startfr - 2]  # GT2D.txt is 1-index
        # print(total_GT)
        # exit()/
        ref_anno = prepareTools.indexSwich(ref_anno)
        # 2-2.ref_img-->examplar(crop as square shape)
        ref_examplar, ref_img = prepareTools.getExamplar(ref_file, ref_anno)
        # 3.img_files: total path of imgs.
        img_files = total_file[startfr - 1:endfr]
        # 4.anno is [x,y,w,h] of each frame
        img_anno = total_GT[startfr - 1:endfr]
        img_anno = prepareTools.indexSwich(img_anno)
        # 5.load img
        img_data = list()
        for i in range(len(img_files)):
            img_org = ops1.read_image(img_files[i])
            img_data.append(img_org)
    ###-----TRACKING----------
        error_curve = list()
        error_total = 0
        error3d_total = 0
        MAE_curve = list()
        seq_result = {}
        for i in range(len(img_data)):
            gt2d = np.array([img_anno[i][0]+img_anno[i][2]/2,
                            img_anno[i][1]+img_anno[i][3]/2])
            print('GT2D:', gt2d)
            img_org = img_data[i]

            if i == 0:  ###first frame center = [c_y,c_x]
                center = np.array([ref_anno[1]+ref_anno[3]/2, ref_anno[0]+ref_anno[2]/2])
            else:
                # ref_anno = total_GT[startfr - 2 + i]  # GT2D.txt is 1-index
                # ref_anno = prepareTools.indexSwich(ref_anno)
                # center = np.array([ref_anno[1] + ref_anno[3] / 2, ref_anno[0] + ref_anno[2] / 2])
                center = np.array([loc2d[1], loc2d[0]])
    #crop the img_org to 120*120,center is last box
            _, sample = ops1.crop_and_resize(
                img_org, center, size = 120,
                out_size= 120,
                border_value= np.mean(img_org, axis=(0, 1)))

            frameNum = i + startfr - 1#0-index
    #AUDIO MEASUREMENT:
            loc, GCFmap = au_observe.au_observ(img_org, DATA, GCC, cam_number, frameNum)
    #NETWORK
            imgRef, img0, img1, img2, auFr = getdata(ref_examplar, sample, GCFmap)
            outputs, total_loss = net(imgRef, img0, img1, img2, auFr)
            #p_in_sample-->p_in_img_org
            output = outputs.detach().cpu().numpy()[0]
            print('output:', output)
            loc2d = np.array([output[0] + center[1] - s_size / 2, output[1] + center[0] - s_size / 2])
            print("loc2d:", loc2d)
            error2d = np.sqrt(np.sum(np.asarray(loc2d - gt2d) ** 2))
            error_curve.append(error2d)
            error_total += error2d
            MAE = error_total / (i + 1)
            MAE_curve.append(MAE)

            cnt += 1
            threshold = ((img_anno[i][2] ** 2 + img_anno[i][3] ** 2) ** 0.5) / 30
            print("threshold:", threshold)
            if error2d <= threshold:
                hit += 1

            print("seq:{} cam:{} sample:{:0>3}/{:0>4} [error2d:{:.4f} MAE:{:.4f} ACC:{:.4f}]".
                  format(sequence, cam_number, i, len(img_files), error2d, MAE, hit / cnt))
            if i == len(img_files) - 1:  # 最后一张
                results[sequence][f'cam{cam_number}']['MAE'] = MAE
                results[sequence][f'cam{cam_number}']['ACC'] = hit / cnt
                results[sequence][f'cam{cam_number}']['error_curve'] = error_curve  # list

print(results)
import json
import pickle
with open(f'{BASE_DIR}/log_auto/{fa_a}_{fa_v}_{fg_a}_{fg_v}_{lg_a}_{lg_v}{"_decay" if decay else ""}_{attention_name}_results.pkl', 'wb') as f:
    pickle.dump(results, f)
# print(json.dumps(results, indent=4))
with open(f'{BASE_DIR}/log_auto/{fa_a}_{fa_v}_{fg_a}_{fg_v}_{lg_a}_{lg_v}{"_decay" if decay else ""}_{attention_name}_results.json', 'w') as f:
    json.dump(results, f, indent=4)


