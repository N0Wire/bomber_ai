import torch
import torch.nn as nn
import torch.nn.functional as F


#simple feed forward q network
#consisting of 2 hidden layers
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


#dueling q network using feed forward layers
class DuelingFFQN(nn.Module):
	def __init__(self, insize, outsize):
		self.outsize = outsize
		super(DuelingFFQN, self).__init__()
		
		#define Network
		self.ll1 = nn.Linear(insize, 1000)
		self.ll2 = nn.Linear(1000, 800)
		
		#now split up in action and value network
		self.a1 = nn.Linear(800, 400)
		self.a2 = nn.Linear(400, outsize)
		
		self.v1 = nn.Linear(800, 400)
		self.v2 = nn.Linear(400, 1)
	
	def forward(self, x):
		x = F.relu(self.ll1(x))
		x = F.relu(self.ll2(x))
		
		a = F.relu(self.a1(x))
		a = F.relu(self.a2(a))
		
		v = F.relu(self.v1(x))
		v = F.relu(self.v2(v))
		
		#now do the aggregation
		#Q(s,a) = V(s) + (A(s,a) - <A>)
		avg = a.mean(1).unsqueeze(1).expand(a.size(0), self.outsize) #calculate average of a and format it to be ready for substraction
		
		x = v + a - avg
		
		return x
