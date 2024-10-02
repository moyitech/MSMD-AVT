import numpy as np
import random
from torch.utils.data import Dataset
import cv2
import pickle

random.seed(1)
class myDataset(Dataset):
    def __init__(self,dataList, refData):
        self.refData = refData
        self.dataList = dataList
        self.minGCF = -0.01
        self.maxGCF = 0.50

    def __getitem__(self, index):
        imgSamplePath= self.dataList[index]
        # img_file = open(imgSamplePath, 'rb')
        # imgData = pickle.load(img_file) # get whole img data
        # refname = imgSamplePath.split('/')[-2]
        # refImg = self.refData[refname]#
        #
        # auSamplePath = imgSamplePath.replace('imgSample', "GCFmap_train")
        # au_file = open(auSamplePath, 'rb')
        # auData = pickle.load(au_file)


        try:
            with open(imgSamplePath, 'rb') as img_file:
                imgData = pickle.load(img_file)  # get whole img data
        except EOFError:
            print(f"EOFError: File {imgSamplePath} is corrupted or incomplete.")
            raise

        refname = imgSamplePath.split('/')[-2]
        refImg = self.refData[refname]

        auSamplePath = imgSamplePath.replace('imgSample', "GCFmap_train")
        try:
            with open(auSamplePath, 'rb') as au_file:
                auData = pickle.load(au_file)
        except EOFError:
            print(f"EOFError: File {auSamplePath} is corrupted or incomplete.")
            raise


        #sample need resize
        sampleImg = imgData['sampleImg']
        segmentImg = imgData['segImg']



        sampleImg0 = cv2.resize(sampleImg, (300, 300)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        sampleImg1 = cv2.resize(sampleImg, (400, 400)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        sampleImg2 = cv2.resize(sampleImg, (500, 500)
                               , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)

        # print(segmentImg.shape)
        segmentImg = cv2.resize(segmentImg, (360, 360)
                                , interpolation=cv2.INTER_NEAREST)
        # 归一化
        segmentImg = segmentImg / 255.0
        segmentImg = np.expand_dims(segmentImg, axis=0)
        # print(segmentImg.shape)
        sampleFace = imgData['sampleFace']
        gt2d = np.array([sampleFace[0] + sampleFace[2] / 2, sampleFace[1] + sampleFace[3] / 2])
        _, GCFmap = auData['GCFmap']  # 3*H*W
        # print(GCFmap)
        # print(GCFmap[1].shape)
        GCF_nor = (GCFmap - self.minGCF) / (self.maxGCF - self.minGCF)
        GCF_nor_expanded = np.expand_dims(GCF_nor, axis=-1)

        # 将新的维度插入到指定位置
        GCF_nor = np.repeat(GCF_nor_expanded, 3, axis=-1)
        # GCF_nor = np.transpose(GCF_nor, (2, 0, 1))  # H*W*3
        # print(GCF_nor.shape)
        reGCFmap = cv2.resize(GCF_nor, (400, 400)
                              , interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
        return refImg, sampleImg0, sampleImg1, sampleImg2, reGCFmap, gt2d, segmentImg

    def __len__(self):
        return len(self.dataList)





