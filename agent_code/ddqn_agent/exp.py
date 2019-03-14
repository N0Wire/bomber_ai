"""
This file contains several functions used for exploration/explotation during training
"""

import numpy as np
import torch

from settings import s

def greedy_epsilon(net, state, epsilon):
	r = np.random.random()
	if r > epsilon:
		#take action proposed by policy-net -> exploitation
		with torch.no_grad():
			return net(state).max(0)[1] #only take the index
	else:
		#choose random action -> exploration
		return np.random.randint(len(s.actions))

def softmax_action(net):
	pass

