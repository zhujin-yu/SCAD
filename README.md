Scene-Conditioned Adaptive Diffusion for Real-World Image Super-Resolution under Heterogeneous Degradations
Official PyTorch implementation of the paper "Scene-Conditioned Adaptive Diffusion for Real-World Image Super-Resolution under Heterogeneous Degradations" .
Code: https://github.com/zhujin-yu/SCAD


Requirements
Python 3.8+
PyTorch 1.12+
CUDA 11.3+ (recommended)


Setup
git clone https://github.com/zhujin-yu/SCAD.git
cd SCAD
torch>=1.12.0
torchvision>=0.13.0
numpy
opencv-python
pillow
tqdm
einops
scipy
scikit-image

 Dataset Preparation：https://pan.baidu.com/s/1OZLAAAlDPLUBzoBxm0pGag?pwd=njfv


python sr_mfe.py -p train -c config/SCAD_train_64_256.json   # train x4
python sr_mfe.py -p val -c config/SCAD_test_64_256.json      # test  x4
python sr_mfe.py -p train -c config/SCAD_train_32_256.json  # train x8
python sr_mfe.py -p val -c config/SCAD_test_32_256.json      # test  x8
python infer.py -p val -c config/SCAD_infer_x4.json      # infer  x4
