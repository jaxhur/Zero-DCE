import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)

RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_supported_image(file_name):
	"""Return True when file_name is a supported training image."""
	return os.path.splitext(file_name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def populate_train_list(lowlight_images_path):


	image_list_lowlight = [
		os.path.join(lowlight_images_path, file_name)
		for file_name in sorted(os.listdir(lowlight_images_path))
		if os.path.isfile(os.path.join(lowlight_images_path, file_name))
		and is_supported_image(file_name)
	]

	train_list = image_list_lowlight

	random.shuffle(train_list)

	return train_list

	

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = populate_train_list(lowlight_images_path) 
		self.size = 256

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


		

	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path).convert("RGB")
		
		data_lowlight = data_lowlight.resize((self.size,self.size), RESAMPLE)
		# Convert to numpy array and normalize
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

