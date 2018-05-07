import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Resnet152(nn.Module):
    def __init__(self, embedding_dim = 512, pretrained = False):
        super(Resnet152, self).__init__()
        self.embedding_dim = embedding_dim
        self.resnet152 = models.resnet152(pretrained=pretrained)
        self.linear = nn.Linear(self.resnet152.fc.in_features, embedding_dim)
        self.resnet152.fc = self.linear
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet152(images)
        # embed = self.batch_norm(embed)
        return embed
