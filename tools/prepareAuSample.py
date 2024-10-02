from tools import ops
import pickle
import glob
import os
import cv2
import numpy as np
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls
from GCF.GCF_extract_stGCF import getGCF

seqList = ['seq03-1p-0000']
datasetPath = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/STNet'
au_observe = getGCF()

for sequence in seqList:
    # load audio data
    audioDATA = ops.loadaudioDATA(sequence, datasetPath)  # 加载音频数据
    GCCdata = ops.loadGCC(sequence, datasetPath)  # 加载GCC数据
    for cam_number in range(3, 4):
        GCC = GCCdata[f'{sequence}_cam{cam_number}']
        DATA = audioDATA[f'{sequence}_cam{cam_number}']
        folderPath = f'{datasetPath}/imgSample/{sequence}_cam{cam_number}/'
        fileList = sorted(glob.glob(folderPath + '*.pkl'))  # 使用 glob 模块列出指定路径下的所有pickle文件，并按文件名排序
        error_curve = list()
        error_total = 0
        MAE_curve = list()

        # 循环遍历每个 pickle 文件
        for i in range(len(fileList)):
            # 打开 pickle 文件并加载数据
            pkl_file = open(fileList[i], 'rb')
            sampleIf = pickle.load(pkl_file)
            # detection = sampleIf['detection']
            imgAnno = sampleIf['imgAnno']  # 获取图像注释信息
            refImg = sampleIf['refImg']
            frameNum = sampleIf['frameNum']  # 获取帧数
            imgPath = sampleIf['imgPath']  # 获取样本图像
            refPath = sampleIf['refPath']  # 获取参考图像
            img = ops.read_image(imgPath)
            # bbox = sampleIf['bbox']
            # # 调用 showRecimg 函数显示样本图像及目标区域
            # ops.showRecimg(img, bbox)
        #get GCFmap, and depth index
            _, GCFmap = au_observe.au_observ(img, DATA, GCC, cam_number, frameNum)
            # GCFmap, depth_ind, gt3d = au_observe.au_observ(sequence, cam_number, frameNum, img, box=imgAnno, spl_box=bbox)
            gcfData = {
                'GCFmap': GCFmap,
                # 'depth_ind': depth_ind,
                # 'GT3D':gt3d,
            }
        ###--- save the imgDataList as {sequence}_sampleList.npz
            folderPath = f'{datasetPath}/GCFmap_train/{sequence}_cam{cam_number}'
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            filename = str(10000 + i)[1:]  # '0000.pkl'
            outputPath = open(f'{folderPath}/{filename}.pkl', 'wb')
            pickle.dump(gcfData, outputPath)  # 将处理后的数据 gcfData 写入pickle文件中
            print(f'save gcfData.pkl for {sequence}_cam{cam_number}_{filename}')
