import torch
import torch.nn as nn
import torch.nn.functional as F


#simple feed forward q network
#consisting of 2 deep layers
class FFQN(nn.Module):
	def __init__(self, insize, outsize):
		super(FFQN, self).__init__()
		
		#define Network here
		self.ll1 = nn.Linear(insize, 1000)
		self.ll2 = nn.Linear(1000, 800)
		self.ll3 = nn.Linear(800, 400)
		self.ll4 = nn.Linear(400, outsize)
	
	def forward(self, x):
		#pass x trough every layer via ReLUs
		x = F.relu(self.ll1(x))
		x = F.relu(self.ll2(x))
		x = F.relu(self.ll3(x))
		x = self.ll4(x)
		
		return x
