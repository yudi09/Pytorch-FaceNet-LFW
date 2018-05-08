import os
import time
import torch
import itertools
import torchvision
import numpy as np
import torch.nn as nn
from Resnet import Resnet
from Alexnet import Alexnet
from Resnet152 import Resnet152
from TripletNet import TripletNet
from DataLoader import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Function
from TripletLoss import triplet_loss
from utils import accuracy, final_label2embeds, gen_triplets, make_minibatches, batch2images_tensor, label2embeds_list2dict


if __name__ == '__main__':
	
	train_dir = 'train'
	dev_dir = 'dev'

	transform = transforms.Compose([transforms.Resize((224, 224)), 
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5),
														(0.5, 0.5, 0.5))
									])

	tic = time.time()
	train_dataloader = DataLoader(train_dir, transform)
	train_data = train_dataloader.gen_data()
	toc = time.time()
	print('train_data loaded %.2f' %((toc-tic)/60))

	tic = time.time()
	dev_dataloader = DataLoader(dev_dir, transform)
	dev_data = dev_dataloader.gen_data()
	toc = time.time()
	print('dev_data loaded %.2f' %((toc-tic)/60))

	embedding_dim = 64
	# model_name = 'alexnet'
	# model_name = 'resnet'
	model_name = 'resnet152'
	if not os.path.exists(model_name):
		os.makedirs(model_name)

	if model_name == 'alexnet':
		cnn = Alexnet(embedding_dim = embedding_dim, pretrained = False)
	elif model_name == 'resnet':
		cnn = Resnet(embedding_dim = embedding_dim, pretrained = False)
	elif model_name == 'resnet152':
		cnn = Resnet152(embedding_dim = embedding_dim, pretrained = False)
	triplet_net = TripletNet(cnn)

	gpu_device = 1
	if torch.cuda.is_available():
		with torch.cuda.device(gpu_device):
			triplet_net.cuda()

	num_epochs = 100000
	minibatch_size = 8
	learning_rate = 1e-4
	params = triplet_net.parameters()
	optimizer = torch.optim.Adam(params, lr = learning_rate)

	for epoch in range(num_epochs):
		triplet_net.train()
		minibatches = make_minibatches(train_data, minibatch_size = minibatch_size, seed = epoch)
		loss = []
		tic = time.time()
		for cur_minibatch in minibatches:
			triplet_net.zero_grad()
			images_tensor = batch2images_tensor(cur_minibatch[0], train_dataloader, gpu_device)
			embeds = triplet_net(images_tensor)
			id2embeds = label2embeds_list2dict(cur_minibatch[0], embeds)
			anchor, positive, negative = gen_triplets(cur_minibatch, id2embeds, embedding_dim, gpu_device)
			if anchor.shape[0] != 0:
				l = triplet_loss(anchor, positive, negative)
				loss.append(l)
				l.backward()
				optimizer.step()
		toc = time.time()

		# print('epoch %d train_loss %f time %.2f mins' 
		# 	 %(epoch, torch.mean(torch.Tensor(loss)), (toc-tic)/60))
		label2embeds = final_label2embeds(triplet_net, train_dataloader, gpu_device)
		
		if (epoch + 1) % 10 == 0:
			torch.save(label2embeds, os.path.join(model_name, 'iter_%d_label2embeds.pkl'%(epoch)))		
			torch.save(triplet_net.state_dict(), os.path.join(model_name, 'iter_%d_triplet_net.pkl'%(epoch)))

		if (epoch+1) % 10 == 0:
			train_acc, num_train = accuracy(train_data, train_dataloader, label2embeds, triplet_net, gpu_device)
			dev_acc, num_dev = accuracy(dev_data, dev_dataloader, label2embeds, triplet_net, gpu_device)
			print('%d %f %f %f' 
			 %(epoch, torch.mean(torch.Tensor(loss)), train_acc/num_train, dev_acc/num_dev))
			
			# train accuracy
			# tic = time.time()
			# train_acc, num_train = accuracy(train_data, train_dataloader, label2embeds, triplet_net, gpu_device)
			# toc = time.time()
			# print('train_acc %f correct %d/%d time %.2f mins' 
			# 		%(train_acc/num_train, train_acc, num_train, (toc-tic)/60))

			# # dev accuracy
			# tic = time.time()
			# dev_acc, num_dev = accuracy(dev_data, dev_dataloader, label2embeds, triplet_net, gpu_device)
			# toc = time.time()
			# print('dev_acc %f correct %d/%d time %.2f mins' 
			# 		%(dev_acc/num_dev, dev_acc, num_dev, (toc-tic)/60))