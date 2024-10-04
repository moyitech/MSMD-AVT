# MSMD-AVT
MSMD-AVT：Multi-Stage Multimodal Distillation for Audio-Visual Speaker Tracking
## Requirements
+ Python3.8
+ PyTorch 2.0
+ Numpy
# Data Preparation
+ AV16.3: the original dataset, available at http://www.glat.info/ma/av16.3/
+ For training the MSMD-AVT, you should prepare the audio-visual samples:
    + tools/prepareAudio.py, prepare_gccphat.py
    + tools/prepareSample.py, prepareAusample
## Descriptions
**Train**
+ To train MSMD-AVT, you need to download seq01, 02, 03and camera parameters from the AV16.3 dataset. Use the preprocessing files provided in the tools for audio and video synchronization, audio preprocessing and prepare audio-visual sample pairs for training
+ After preparing the dataset and training samples, set the path of the correct image samples and GCF samples path in tain_auto.py and run it.

**Tracking**
+ tracking sequences are from seq08,11,12 in AV16.3 dataset.
+ Run test_auto.py to track.
