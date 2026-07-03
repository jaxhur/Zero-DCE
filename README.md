原始仓库：https://github.com/Li-Chongyi/Zero-DCE

> 后续同一个作者还发了[Zero-DCE++](https://ieeexplore.ieee.org/document/9369102)，进行了改进

**1、环境配置**

```bash
# conda create --name zerodce_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch
# 原论文的太老了，升级了版本
conda create -n zerodce_env python=3.8 -y
conda activate zerodce_env
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 opencv-python --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**2、准备数据集**

- **测试数据**：低照度图片放入 `data/test_data/LIME` 等子文件夹（项目自带了一些测试集）。
- **训练数据**：下载[训练数据](https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view) ，并解压到 `data/` 目录
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

**3、测试模型，初步跑通**

项目提供了一个预训练模型 `snapshots/Epoch99.pth`。直接运行：
```bash
python lowlight_test.py 
```
自动读取 `data/test_data/` 下所有子目录中的图片，进行增强后将结果保存在 `data/result/` 中

**4、训练模型**

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

# 问题

TODO：

- PSNR/SSIM的evaluate
- Loss曲线、Loss保存到csv文件
- Log打印和输出文件

