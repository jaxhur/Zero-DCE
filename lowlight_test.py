import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import time
import dataloader
import model
import numpy as np
from PIL import Image
import glob
import time


 
def lowlight(image_path, dce_net, device, input_root, output_root):
	"""  
	单张图像低光增强
	"""
	data_lowlight = Image.open(image_path)

	# 图像预处理
	data_lowlight = (np.asarray(data_lowlight)/255.0) # 0~255 归一化到 0~1
	data_lowlight = torch.from_numpy(data_lowlight).float() # 转换为 PyTorch 张量
	data_lowlight = data_lowlight.permute(2,0,1) # HWC to CHW(PyTorch卷积网络输入格式是 CHW)
	data_lowlight = data_lowlight.to(device).unsqueeze(0) # BCHW(1,C,H,W) 

	# 模型增强图像
	start = time.time()
	_,enhanced_image,_ = dce_net(data_lowlight)
	end_time = (time.time() - start)
	print(f"image_process_time: {end_time:.5f} s")

	rel_path = os.path.relpath(image_path, input_root)
	result_path = os.path.join(output_root, rel_path)
	result_dir = os.path.dirname(result_path)
	os.makedirs(result_dir, exist_ok=True)

	torchvision.utils.save_image(enhanced_image, result_path)

if __name__ == '__main__':
	""" 
	测试
	"""
	with torch.no_grad():
		input_root = os.path.join('data', 'test_data')
		output_root = os.path.join('data', 'result')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		dce_net = model.enhance_net_nopool().to(device)
		dce_net.load_state_dict(torch.load(os.path.join('snapshots', 'Epoch99.pth'),
									  map_location=device))
		dce_net.eval()
	
		file_list = os.listdir(input_root)
		for file_name in file_list:
			test_list = glob.glob(os.path.join(input_root, file_name, "*"))
			for image in test_list:
				print(f"Processing image: {image}")
				lowlight(image, dce_net, device, input_root, output_root)

		

