"""
This file contains several functions used for exploration/explotation during training
"""

import numpy as np
import torch
import torch.nn.functional as F

from settings import s

def random_action():
	return np.random.randint(len(s.actions))

def greedy_epsilon(net, state, epsilon):
	r = np.random.random()
	if r > epsilon:
		#take action proposed by policy-net -> exploitation
		with torch.no_grad():
			return net(state).max(0)[1] #only take the index
	else:
		#choose random action -> exploration
		return np.random.randint(len(s.actions))

def boltzmann(net, state, temp=0.5):
	with torch.no_grad():
		pvals = F.softmax(net(state)/temp, dim=0)
		return pvals.max(0)[1]

