import torch
import torch.nn as nn


class TripletLoss(nn.Module):
	def __init__(self, alpha = 0.2):
		super(TripletLoss, self).__init__()
		self.alpha = alpha
			
	def forward(self, anchor, positive, negative):
		alpha = self.alpha
		pos_dist = anchor - positive
		pos_dist = torch.pow(pos_dist, 2).sum(dim=1)
		neg_dist = anchor - negative
		neg_dist = torch.pow(neg_dist, 2).sum(dim=1)
		basic_loss = pos_dist - neg_dist + alpha
		# loss = torch.clamp(basic_loss, min=0.0).sum()
		relu = nn.ReLU()
		loss = relu(basic_loss)
		return loss.mean()

def triplet_loss(anchor, positive, negative, alpha=0.2):
	return TripletLoss(alpha)(anchor, positive, negative)