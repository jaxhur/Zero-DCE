
## Environment Setup (Updated for PyTorch + CUDA 11.3)

```bash
# 1. Create conda environment
conda create --name zerodce_env python=3.8 -y
conda activate zerodce_env

# 2. Install PyTorch 1.11.0 with CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

# 3. Install other dependencies
pip install opencv-python Pillow numpy
```

### Test: 


cd Zero-DCE_code
```
python lowlight_test.py 
```
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.

### Train: 
1) cd Zero-DCE_code

2) download the training data <a href="https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA">baidu cloud [password: 1234]</a>

3) unzip and put the  downloaded "train_data" folder to "data" folder
```
python lowlight_train.py 
```
##  License
The code is made available for academic research purpose only. Under Attribution-NonCommercial 4.0 International License.


## Bibtex

```
@inproceedings{Zero-DCE,
 author = {Guo, Chunle Guo and Li, Chongyi and Guo, Jichang and Loy, Chen Change and Hou, Junhui and Kwong, Sam and Cong, Runmin},
 title = {Zero-reference deep curve estimation for low-light image enhancement},
 booktitle = {Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR)},
 pages    = {1780-1789},
 month = {June},
 year = {2020}
}
```

(Full paper: http://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)

## Contact
If you have any questions, please contact Chongyi Li at lichongyi25@gmail.com or Chunle Guo at guochunle@tju.edu.cn.

## TensorFlow Version 
Thanks tuvovan (vovantu.hust@gmail.com) who re-produces our code by TF. The results of TF version look similar with our Pytorch version. But I do not have enough time to check the details.
https://github.com/tuvovan/Zero_DCE_TF
