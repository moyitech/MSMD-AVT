import os
import glob
import random
import numpy as np
from tools import ops
from tools import prepareTools
import pickle
from tqdm import tqdm

random.seed(1)
# #------------------set seqList-----------------------------------
# seqList = ['seq12-1p-0100'] #'seq01-1p-0000','seq02-1p-0000'
# seqList = ['seq08-1p-0100', 'seq11-1p-0100', 'seq12-1p-0100'] #'seq01-1p-0000','seq02-1p-0000'
# seqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000'] #'seq01-1p-0000','seq02-1p-0000'
seqList = ['seq03-1p-0000'] #'seq01-1p-0000','seq02-1p-0000'
datasetPath_vi = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/AV163'
datasetPath = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/STNet'#save path

for sequence in seqList:
    for cam_number in range(3, 4):
        print(sequence, cam_number)
        startfr, endfr = ops.getSE(sequence, cam_number)#1-index
        seq_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/img/'
        segment_dir = f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/segment/'
        print(startfr, endfr)
    #1.ref(path) is from the previous frame of the startfr:startfr - 2 (0-index)
        # print(sorted(glob.glob(seq_dir + '*.jpg')))
        # print(sorted(glob.glob(segment_dir + '*.jpg')))
        files = [os.path.join(seq_dir, file) for file in os.listdir(seq_dir)]
        # segment_files = [os.path.join(segment_dir, file) for file in os.listdir(segment_dir)]

        # 遍历文件列表，输出文件名
        # for file in files:
        #     print(file)
        ref_file = sorted(glob.glob(seq_dir + '*.jpg'))[startfr - 2]
    #2.ref_anno
        ref_anno = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                              f'{sequence}_cam{cam_number}_GT2D.txt')[startfr - 2]#GT2D.txt is 1-index
        ref_anno = prepareTools.indexSwich(ref_anno)#"1-index to 0-index,[x,y,w,h]"
    #2-2.ref_img-->examplar(crop as square shape)
        ref_examplar, ref_img = prepareTools.getExamplar(ref_file, ref_anno)
    #3.img_files: total path of imgs.
        img_files = sorted(glob.glob(seq_dir + '*.jpg'))[startfr - 1:endfr]
        segment_files = sorted(glob.glob(segment_dir + '*.jpg'))[startfr - 1:endfr]
        print(len(img_files), len(segment_files))
    #4.anno is [x,y,w,h] of each frame
        img_anno = np.loadtxt(f'{datasetPath_vi}/{sequence}/{sequence}_cam{cam_number}_jpg/'
                              f'{sequence}_cam{cam_number}_GT2D.txt')[startfr - 1:endfr]
        img_anno = prepareTools.indexSwich(img_anno)#"1-index to 0-index,[x,y,w,h]"
        for i in tqdm(range(len(img_files))):
            img_org = ops.read_image(img_files[i])
            seg_img = ops.read_segment_image(segment_files[i])
            gtbox = img_anno[i]
            # ops.showRecimg(img_org, gtbox)
    #5.random crop sample from img_org
            sample_img, sample_img_box, sample_face = prepareTools.getSample(img_org, gtbox)
            imgData = {
                'seqName': sequence,
                'camNum': cam_number,#123
                'frameNum': i + startfr - 1,  # 0-index, start from 'startfr - 1'(0-index) for each camare
                'refPath': ref_file,
                'refAnno': ref_anno,# 0-index
                'imgPath': img_files[i],
                'imgAnno': img_anno[i],# 0-index
                'sampleImgBox': sample_img_box,  # sample_img_box for square_img (square coordinate)
                'sampleFace': sample_face,  # face box in sample_img (sample coordinate)
                'refImg': ref_img,
                'sampleImg': sample_img,
                'refExamplar': ref_examplar,
                'segImg': seg_img
            }
        ##### # save the imgDataList as {sequence}_sampleList.npz
            folderPath = f'{datasetPath}/imgSample/{sequence}_cam{cam_number}'
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            filename = str(10000+i)[1:]#'0000.pkl'
            outputPath = open(f'{folderPath}/{filename}.pkl', 'wb')
            pickle.dump(imgData, outputPath)
            # print(f'save imgsample.pkl for {sequence}_cam{cam_number}_{filename}')

