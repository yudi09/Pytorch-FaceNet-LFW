import os
import torch
from PIL import Image
from torchvision import transforms


class DataLoader():
	def __init__(self, dir_path, transform):
		self.images_dict = {}
		self.id2image = {}
		self.labels = None
		self.dir_path = dir_path
		self.transform = transform
		self.load_images()
	
	def load_images(self):
		# returns labels/names list
		self.labels = os.listdir(self.dir_path)
		for label in self.labels:
			path = os.path.join(self.dir_path, label)
			images = os.listdir(path)
			self.images_dict[label] = images
			for image_id in images:
				img_path = os.path.join(path, image_id)
				self.id2image[image_id] = self.transform(Image.open(img_path))
	
	def gen_data(self):
		labels = []
		image_ids = []
		for label, images in self.images_dict.items():
			num_images = len(images)
			labels.extend([label] * num_images)
			image_ids.extend(images)
		return image_ids, labels
		
	def get_image(self, image_id):
		return self.id2image[image_id]
