import os
import pickle
from tools.prepareClass import InitGCF
from tools.prepareClass import DataMain
from tools.prepareClass import configGCF
from tools.prepareClass import camCls

if __name__ == '__main__':
    # seqList = ['seq01-1p-0000', 'seq02-1p-0000', 'seq03-1p-0000']
    # seqList = ['seq08-1p-0100']
    seqList = ['seq08-1p-0100', 'seq11-1p-0100', 'seq12-1p-0100']
    datasetPath_au = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/AV163'
    datasetPath_save = '/amax/tyut/user/wzy/workspace/KD-DIGLIM/STNet'

    for sequence in seqList:
        audioDataDic = {}
        for cam_number in range(1, 4):
            DATA, CFGforGCF = InitGCF(datasetPath_au, sequence, cam_number)
            audioData = {f'{sequence}_cam{cam_number}': DATA}
            audioDataDic.update(audioData)

        folderPath = f'{datasetPath_save}/audio/{sequence}'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        output = open(f'{folderPath}/{sequence}_audio.pkl', 'wb')
        pickle.dump(audioDataDic, output)
        print(f'save audio.npz for {sequence}')
