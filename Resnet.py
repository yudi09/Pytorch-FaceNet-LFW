import os
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms


class Resnet(nn.Module):
    def __init__(self, embedding_dim=256, pretrained = False):
        super(Resnet, self).__init__()
        self.embedding_dim = embedding_dim
        self.resnet18 = models.resnet18(pretrained=pretrained)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, embedding_dim)
        # self.batch_norm = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        self.init_weights()
    
    def init_weights(self):
        self.resnet18.fc.weight.data.normal_(0.0, 0.02)
        self.resnet18.fc.bias.data.fill_(0)
    
    def forward(self, images):
        embed = self.resnet18(images)
        # embed = self.batch_norm(embed)
        return embed
