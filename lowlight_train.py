import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
	"""  
	权重初始化
	卷积层权重:均值为0、标准差为0.02的正态分布
	批归一化层:均值为1、标准差为0.02的正态分布，偏置项初始化为0。
	"""
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)



def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'
	# 定义模型
	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.apply(weights_init)
	if config.load_pretrain == True:
		DCE_net.load_state_dict(torch.load(config.pretrain_dir))
	# 定义dataloader
	train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
	train_loader = torch.utils.data.DataLoader(train_dataset, 
											batch_size=config.train_batch_size, 
											shuffle=True, 
											num_workers=config.num_workers, 
											pin_memory=True)

	# 定义损失函数
	L_color = Myloss.L_color()
	L_spa = Myloss.L_spa()
	L_exp = Myloss.L_exp(16,0.6)
	L_TV = Myloss.L_TV()
	# 定义优化器
	optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	DCE_net.train()
	for epoch in range(config.num_epochs):
		for iteration, img_lowlight in enumerate(train_loader):

			img_lowlight = img_lowlight.cuda()
			optimizer.zero_grad()
			# 模型增强图像
			enhanced_image_1,enhanced_image,A  = DCE_net(img_lowlight)
			# 计算损失
			Loss_TV = 200*L_TV(A)
			loss_spa = torch.mean(L_spa(img_lowlight, enhanced_image))
			loss_col = 5*torch.mean(L_color(enhanced_image))
			loss_exp = 10*torch.mean(L_exp(enhanced_image))
			loss =  Loss_TV + loss_spa + loss_col + loss_exp
			# 反向传播
			loss.backward()
			torch.nn.utils.clip_grad_norm_(DCE_net.parameters(),
								  config.grad_clip_norm) # 梯度裁剪，防止梯度爆炸
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0: # 每10次迭代打印一次损失
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0: # 每1次epoch保存一次模型
				snapshot_path = os.path.join(config.snapshots_folder, "Epoch" + str(epoch) + '.pth')
				torch.save(DCE_net.state_dict(), snapshot_path) 		



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=200)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=4)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default= False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch99.pth")
	config = parser.parse_args()
	# 创建保存模型的文件夹
	os.makedirs(config.snapshots_folder, exist_ok=True)

	train(config)








	
