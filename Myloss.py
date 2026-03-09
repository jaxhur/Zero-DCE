import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np


class L_color(nn.Module):
    """  
    计算RGB三通道全局均值差异，抑制偏色
    """

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)  # 每个通道的全局均值(B,3,1,1)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)  # 拆出RGB三个通道的全局均值(B,1,1,1)
        # 计算RGB三通道之间的全局均值差异
        Drg = torch.pow(mr-mg, 2)
        Drb = torch.pow(mr-mb, 2)
        Dgb = torch.pow(mb-mg, 2)
        # 每个batch一个标量损失
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) +
                      torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    """  
    比较增强前后在局部方向梯度(left/right/up/down)上的结构差异，尽量保持结构关系。
    """

    def __init__(self):
        super(L_spa, self).__init__()
        # 定义四个卷积核，分别计算图像在四个方向上的一阶差分
        kernel_left = torch.FloatTensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(
            data=kernel_left, requires_grad=False)  # 固定算子，不参与训练
        self.weight_right = nn.Parameter(
            data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        # 对输入图像、增强图像在通道维度求均值，得到单通道图
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        # 对单通道图进行平均池化，降低分辨率，扩大感受野
        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        # 未使用
        weight_diff = torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor(
            [0.3]).cuda(), torch.FloatTensor([0]).cuda()), torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(
            enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool-org_pool)

        # 分别计算增强前后图像在四个方向上的一阶差分
        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        # 计算增强前后在四个方向上的结构差异
        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)

        return E


class L_exp(nn.Module):
    """  
    曝光控制损失:增强后图像的亮度接近一个目标值。
    """

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size) # 16x16的patch，分块平均，得到局部区域的平均亮度
        self.mean_val = mean_val # 指定亮度0.6

    def forward(self, x):

        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True) # 变为单通道图
        mean = self.pool(x) # 计算每个patch的平均亮度
        # 局部亮度与目标亮度的均方误差
        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    """  
    总变分损失:增强图像的增强曲线尽量平滑，避免过度增强引入噪点。
    """
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        # 水平方向和垂直方向上的相邻像素对的数量
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum() # 竖直方向相邻像素差的平方和
        w_tv = torch.pow((x[:, :, :, 1:]-x[:, :, :, :w_x-1]), 2).sum() # 水平方向相邻像素差的平方和
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size # 横纵两个方向的差异加起来，并做归一化


# 下面两个loss并没有使用
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) +
                      torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
