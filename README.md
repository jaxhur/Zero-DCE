原始仓库：https://github.com/Li-Chongyi/Zero-DCE

> 后续同一个作者还发了[Zero-DCE++](https://ieeexplore.ieee.org/document/9369102)，进行了改进



# 原论文

**1、下载代码**

```shell
git clone https://github.com/jaxhur/Zero-DCE.git
```

**2、环境配置**

```bash
# conda create --name zerodce_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch
# 原论文的太老了，升级了版本
cd /workspace/Zero-DCE

conda create -n zerodce python=3.8 -y
conda activate zerodce
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 opencv-python --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**3、准备数据集**

- **测试数据**：低照度图片放入 `data/test_data/LIME` 等子文件夹（项目自带了一些测试集）。
- **训练数据**：下载[训练数据](https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view)，并解压到 `data/` 目录
  -  SICE 数据集中的多曝光图像，训练图像共 2422 张(实际只有2002张)，验证集是剩余部分
  - 200个epoch，每个epoch大概250个batch

- 代码阅读顺序
  - lowlight_test.py
  - model.py
  - MyLoss.py
  - lowlight_train.py
  - dataloader.py


```

├── data
│   ├── test_data # testing data. You can make a new folder for your testing data, like LIME, MEF, and NPE.
│   │   ├── LIME 
│   │   └── MEF
│   │   └── NPE
│   └── train_data 
├── lowlight_test.py # testing code
├── lowlight_train.py # training code
├── MyLoss.py
├── model.py # Zero-DEC network
├── dataloader.py
├── snapshots
│   ├── Epoch99.pth #  A pre-trained snapshot (Epoch99.pth)
```

**4、测试模型，初步跑通**

项目提供了一个预训练模型 `snapshots/Epoch99.pth`。直接运行：
```bash
python lowlight_test.py 
```
自动读取 `data/test_data/` 下所有子目录中的图片，进行增强后将结果保存在 `data/result/` 中

**5、训练模型**

```bash
python lowlight_train.py 
```
网络会自动开始训练，模型很小，训练速度很快

```
我要在lolv1和lolv2（real、syn）数据集上复现这个项目，训练模型、测试，输出模型在测试集上的PSNR、SSIM、LPIPS，以及模型的参数量Params(M)、TFLOPs

我要在lolv1和lolv2（real、syn）数据集上复现这个项目，训练模型、测试（输出模型在测试集上的PSNR、SSIM、LPIPS以及模型增强后的图片），以及输出训练完成后模型的参数量Params(M)、TFLOPs(G)，当前项目是不是不是在这个两个数据集上训练的？是不是并没有输出模型在测试集上的PSNR、SSIM、LPIPS，以及模型的参数量Params(M)、TFLOPs(G)；你给我修改一下
```

**第5步：消融实验**

- 损失函数的消融分析
  - $L_{spa}$：保持相邻区域的差异性
  - $L_{color}$：矫正颜色偏差
  - $L_{exp}$：控制亮度
  - $L_{TV}$：维持相邻像素的平滑，防止出现生硬的边界
  - **具体操作**：修改 lowlight_train.py
    - 去掉 $L_{spa}$ 训练一个模型，观察模型结果是否会丢失空间结构的纹理。
    - 去掉 $L_{TV}$ 训练一个模型，观察图片是否会出现网格状或不自然的分块伪影。
    - 。。。
- 网络结构的消融分析：DCENet默认迭代 8 次
  - **迭代次数**
  - **网络深度**



# LOLv1

```shell
git clone https://github.com/jaxhur/Zero-DCE.git
```

环境：

```shell
# conda create --name zerodce_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch
# 原论文的太老了，升级了版本
cd /workspace/Zero-DCE

conda create -n zerodce python=3.8 -y
conda activate zerodce
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 opencv-python --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

数据：

```shell
mkdir data

python3 -m pip install -U gdown
apt install -y unzip


cd ./data
# LOL-v1
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z"
# LOL-v2原始
# gdown "https://drive.google.com/uc?id=1dzLJFz0svHXYHvAe-Tl52miChhF4BXXE"
# LOL-v2重命名
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP"

unzip LOL-v1.zip -d LOL-v1
unzip LOL-v2-renamed.zip -d LOL-v2
```

训练：

```shell
python lowlight_train.py --lowlight_images_path data/LOL-v1/our485/low --snapshots_folder snapshots/lolv1
```

默认训练 `200` epoch：

```
snapshots\lolv1\Epoch199.pth
snapshots\lolv2_syn\Epoch199.pth
snapshots\lolv2_real\Epoch199.pth
```

测试

- PSNR：14.279342
- SSIM：0.509626
- LPIPS：0.430033
- Params：0.079416
- FLOPS：10.380902
- flops_input_size：1,3,256,256

```
pip install lpips

python lowlight_test_lol.py --low_dir data/LOL-v1/eval15/low --gt_dir data/LOL-v1/eval15/high --weights snapshots/lolv1/Epoch199.pth --output_dir data/result_lol/lolv1 --dataset_name LOL-v1 --csv_path data/result_lol/lolv1_summary.csv --per_image_csv data/result_lol/lolv1_per_image.csv --flops_input_size 1,3,256,256
```





# LOLv2-real

训练：200epoch

```
python lowlight_train.py --lowlight_images_path data/LOL-v2/Synthetic/Train/Low --snapshots_folder snapshots/lolv2_syn
```

测试：

- PSNR：12.398917
- SSIM：0.447077
- LPIPS：0.487163
- Params：0.079416
- FLOPS：10.380902
- flops_input_size：1,3,256,256

```
python lowlight_test_lol.py --low_dir data/LOL-v2/Synthetic/Test/Low --gt_dir data/LOL-v2/Synthetic/Test/Normal --weights snapshots/lolv2_syn/Epoch199.pth --output_dir data/result_lol/lolv2_syn --dataset_name LOL-v2-syn --csv_path data/result_lol/lolv2_syn_summary.csv --per_image_csv data/result_lol/lolv2_syn_per_image.csv --flops_input_size 1,3,256,256
```



# LOLv2-syn

训练：200epoch

```
python lowlight_train.py --lowlight_images_path data/LOL-v2/Real_captured/Train/Low --snapshots_folder snapshots/lolv2_real
```

测试：

- PSNR：16.402332
- SSIM：0.807187
- LPIPS：0.217667
- Params：0.079416
- FLOPS：10.380902
- flops_input_size：1,3,256,256

```
python lowlight_test_lol.py --low_dir data/LOL-v2/Real_captured/Test/Low --gt_dir data/LOL-v2/Real_captured/Test/Normal --weights snapshots/lolv2_real/Epoch199.pth --output_dir data/result_lol/lolv2_real --dataset_name LOL-v2-real --csv_path data/result_lol/lolv2_real_summary.csv --per_image_csv data/result_lol/lolv2_real_per_image.csv --flops_input_size 1,3,256,256
```

