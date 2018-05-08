import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):
	def __init__(self, cnn):
		super(TripletNet, self).__init__()
		self.embedding = cnn

	# def forward(self, images_tensor, minibatch_X):
	def forward(self, images_tensor):
		embeds = self.embedding(images_tensor)
		# id2embeds = {}
		# minibatch_size = len(minibatch_X)
		# for i in range(minibatch_size):
		# 	x = minibatch_X[i]
		# 	id2embeds[x] = embeds[i, :]
		# return id2embeds
		return embeds
