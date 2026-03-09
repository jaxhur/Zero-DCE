https://github.com/Li-Chongyi/Zero-DCE

**第一步：环境配置**

```bash
# conda create --name zerodce_env opencv pytorch==1.0.0 torchvision==0.2.1 cuda100 python=3.7 -c pytorch
# 原论文的太老了，升级了版本
conda create -n zerodce_env python=3.8 -y
conda activate zerodce_env
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 opencv-python --extra-index-url https://download.pytorch.org/whl/cu113 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**第二步：准备数据集**

- **测试数据**：低照度图片放入 `data/test_data/LIME` 等子文件夹（项目自带了一些测试集）。
- **训练数据**：下载[训练数据](https://drive.google.com/file/d/1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3/view) ，并解压到 `data/` 目录
  -  SICE 数据集中的多曝光图像，训练图像共 2422 张(实际只有2002张)，验证集是剩余部分
  - 200个epoch，每个epoch大概250个batch

- 阅读顺序
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

**第三步：测试模型，初步跑通**

项目提供了一个预训练模型 `snapshots/Epoch99.pth`。直接运行：
```bash
python lowlight_test.py 
```
自动读取 `data/test_data/` 下所有子目录中的图片，进行增强后将结果保存在 `data/result/` 中

**第四步：训练模型**

```bash
python lowlight_train.py 
```
网络会自动开始训练，模型很小，训练速度很快





## 4消融实验

消融实验用于证明你论文（或当前模型）提出的每一个组件（如损失函数、网络块）都是有用的。在复现 Zero-DCE 时，主要可以进行以下两个方面的消融：

### 4.1 损失函数的消融分析
Zero-DCE 使用了 4 种无参考的损失函数（见 [Myloss.py](file:///d:/%E7%A0%94%E7%A9%B6%E7%94%9F%E7%A7%91%E7%A0%94/02_%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E4%BD%8E%E7%85%A7%E5%BA%A6%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/Zero-DCE/Myloss.py) 和 [lowlight_train.py](file:///d:/%E7%A0%94%E7%A9%B6%E7%94%9F%E7%A7%91%E7%A0%94/02_%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E4%BD%8E%E7%85%A7%E5%BA%A6%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/Zero-DCE/lowlight_train.py) 中的 [loss](file:///d:/%E7%A0%94%E7%A9%B6%E7%94%9F%E7%A7%91%E7%A0%94/02_%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E4%BD%8E%E7%85%A7%E5%BA%A6%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/Zero-DCE/Myloss.py#125-158) 组合）：
- **Spatial Consistency Loss ($L_{spa}$)**：保持相邻区域的差异性。
- **Color Constancy Loss ($L_{color}$)**：矫正颜色偏差。
- **Exposure Control Loss ($L_{exp}$)**：控制曝光水平。
- **Illumination Smoothness Loss ($L_{TV}$)**：维持相邻像素的平滑，防止出现生硬的边界。

**具体操作**：
修改 [lowlight_train.py](file:///d:/%E7%A0%94%E7%A9%B6%E7%94%9F%E7%A7%91%E7%A0%94/02_%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E4%BD%8E%E7%85%A7%E5%BA%A6%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/Zero-DCE/lowlight_train.py) 第 72 行的 Loss 计算代码。例如：
- 去掉 $L_{spa}$ 训练一个模型，观察模型结果是否会丢失空间结构的纹理。
- 去掉 $L_{TV}$ 训练一个模型，观察图片是否会出现网格状或不自然的分块伪影。
将这些有缺陷的图片和结合所有 Loss 训练出的完美图片放在一起，即完成了损失函数的消融实验。

### 4.2 网络结构的消融分析
Zero-DCE 是通过迭代估计曲线来实现增强的，默认迭代 8 次（见 [model.py](file:///d:/%E7%A0%94%E7%A9%B6%E7%94%9F%E7%A7%91%E7%A0%94/02_%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E4%BD%8E%E7%85%A7%E5%BA%A6%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/Zero-DCE/model.py) 中 `r1` 到 `r8` 的计算逻辑）。
- **迭代次数的影响**：你可以修改 [model.py](file:///d:/%E7%A0%94%E7%A9%B6%E7%94%9F%E7%A7%91%E7%A0%94/02_%E8%AE%BA%E6%96%87%E5%A4%8D%E7%8E%B0/%E4%BD%8E%E7%85%A7%E5%BA%A6%E5%9B%BE%E5%83%8F%E5%A2%9E%E5%BC%BA/Zero-DCE/model.py) 中的代码，将循环迭代次数（或曲线的阶数）从 8 改为 1，或者 4，或者更高的 16。
- 测试在不同迭代次数下的效果变化与速度变化，绘制曲线图，从而证明选择 8 次迭代是最佳的一个 Trade-off（权衡）。
