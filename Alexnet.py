import torch
import torchvision
import torch.nn as nn
import torchvision.models as models


class Alexnet(nn.Module):
	def __init__(self, embedding_dim = 32, pretrained = False):
		super(Alexnet, self).__init__()
		self.embedding_dim = embedding_dim
		self.alexnet = models.alexnet(pretrained=pretrained)
		in_features = self.alexnet.classifier[6].in_features
		self.linear = nn.Linear(in_features, embedding_dim)
		self.alexnet.classifier[6] = self.linear
		self.init_weights()

	def init_weights(self):
		self.linear.weight.data.normal_(0.0, 0.02)
		self.linear.bias.data.fill_(0)

	def forward(self, images):
		embed = self.alexnet(images)
		return embed