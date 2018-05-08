import os
import math
import torch
import itertools
import torchvision
from torch.autograd import Variable
from TripletNet import TripletNet


def shuffle_data(data, seed = 0):
	image_ids, labels = data
	shuffled_image_ids = []
	shuffled_labels = []
	num_images = len(image_ids)
	torch.manual_seed(seed)
	perm = list(torch.randperm(num_images))
	for i in range(num_images):
		shuffled_image_ids.append(image_ids[perm[i]])
		shuffled_labels.append(labels[perm[i]])
	return shuffled_image_ids, shuffled_labels

def make_minibatches(data, minibatch_size = 16,  seed = 0, shuffle = 'random'):
	X, Y = data
	m = len(X)
	minibatches = []
	if shuffle == 'sequential':
		shuffled_X, shuffled_Y = X, Y

	elif shuffle == 'random':
		shuffled_X, shuffled_Y = shuffle_data(data, seed = seed)

	num_complete_minibatches = math.floor(m/minibatch_size)
	for k in range(0, num_complete_minibatches):
		minibatch_X = shuffled_X[k * minibatch_size : k * minibatch_size + minibatch_size]
		minibatch_Y = shuffled_Y[k * minibatch_size : k * minibatch_size + minibatch_size]
		minibatches.append((minibatch_X, minibatch_Y))

	rem_size = m - num_complete_minibatches * minibatch_size
	if m % minibatch_size != 0:
		minibatch_X = shuffled_X[num_complete_minibatches * minibatch_size : m]
		minibatch_Y = shuffled_Y[num_complete_minibatches * minibatch_size : m]
		minibatches.append((minibatch_X, minibatch_Y))

	return minibatches

def batch2images_tensor(minibatch_X, dataloader, gpu_device):
	minibatch_size = len(minibatch_X)
	images_tensor = torch.zeros(minibatch_size, 3, 224, 224)
	for i in range(minibatch_size):
		x = minibatch_X[i]
		x_image = dataloader.get_image(x)
		images_tensor[i, :, :, :] = x_image
	images_tensor = Variable(images_tensor)
	if torch.cuda.is_available():
		with torch.cuda.device(gpu_device):
			images_tensor = images_tensor.cuda()
	return images_tensor

def gen_triplets(minibatch, id2embeds, embedding_dim, gpu_device, mode = 'all'):
	X, Y = minibatch
	Y_prod = itertools.product(Y, repeat=3)
	X_prod = itertools.product(X, repeat=3)
	triplet = []
	for x, y  in zip(X_prod, Y_prod):
		xa, xp, xn = x
		ya, yp, yn = y
		if (ya == yp) and (ya!=yn) and (xa!=xp):
			triplet.append((xa, xp, xn))
	
	num_triplets = len(triplet)
	anchor = torch.zeros(num_triplets, embedding_dim)
	positive = torch.zeros(num_triplets, embedding_dim)
	negative = torch.zeros(num_triplets, embedding_dim)
	if torch.cuda.is_available():
		with torch.cuda.device(gpu_device):
			anchor = anchor.cuda()
			positive = positive.cuda()
			negative = negative.cuda()

	for i in range(num_triplets):
		xa, xp, xn = triplet[i]
		anchor[i, :] = id2embeds[xa]
		positive[i, :] = id2embeds[xp]
		negative[i, :] = id2embeds[xn]
		
	return anchor, positive, negative

def label2embeds_list2dict(labels_list, embeds):
	label2embeds = {}
	num_labels = len(labels_list)
	for i in range(num_labels):
		label = labels_list[i]
		label2embeds[label] = embeds[i, :]
	return label2embeds

def final_label2embeds(triplet_net, train_dataloader, gpu_device):
	labels_list = []
	image_ids = []
	for label, images_list in train_dataloader.images_dict.items():
		image_ids.append(images_list[0])
		labels_list.append(label)

	images_tensor = batch2images_tensor(image_ids, train_dataloader, gpu_device)
	with torch.no_grad():
		embeds = triplet_net.embedding(images_tensor)
	label2embeds = label2embeds_list2dict(labels_list, embeds)
	return label2embeds

def who_is_it(label2embeds, embed):
	labels = []
	num_labels = len(label2embeds)
	embedding_dim = embed.shape[0]
	embeds = torch.zeros(num_labels, embedding_dim)
	i = 0
	for label, cur_embed in label2embeds.items():
		labels.append(label)
		embeds[i, :] = cur_embed
		i += 1
	dist = torch.pow(embeds - embed, 2).sum(dim = 1)
	index = torch.argmin(dist).tolist()
	return labels[index]

def accuracy(data, dataloader, label2embeds, triplet_net, gpu_device):
	image_ids, Y = data
	num_data = len(Y)
	embedding_dim = triplet_net.embedding.embedding_dim
	embeds = torch.zeros(num_data, embedding_dim)
	minibatch_size = 32
	minibatches = make_minibatches(data, minibatch_size = minibatch_size,  seed = 0, shuffle = 'sequential')
	start = 0
	end = 0
	for cur_minibatch in minibatches:
		minibatch_X, _ = cur_minibatch
		cur_minibatch_size = len(minibatch_X)
		end += cur_minibatch_size
		images_tensor = batch2images_tensor(minibatch_X, dataloader, gpu_device)
		with torch.no_grad():
			embeds[start:end, :] = triplet_net(images_tensor)
		start = end

	acc = 0
	# pred = []

	for i in range(num_data):
		embed = embeds[i]
		target_label = Y[i]
		predicted_label = who_is_it(label2embeds, embed)
		if predicted_label == target_label:
			acc += 1
			# pred.append(predicted_label)
	return acc, num_data
