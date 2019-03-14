"""
This file contains code for the experience replay buffer
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from settings import s

def pack_scene(data):
	"""
	Collect data from current game_state and format data
	into one big vector which then can be used to plan actions
	"""
	#arena map
	arena = np.array(data["arena"])
	player_pos = np.array(data["self"][0:2])
	
	player_canbomb = np.array([data["self"][3]])
	
	opponents_pos = np.zeros((s.max_agents, 2))
	opponents_pos = np.array(data["others"])
	
	bombmap = np.empty(arena.shape)
	bombmap.fill(-1) #-1 means no bomb
	bombs = np.array(data["bombs"])
	if bombs.shape[0] > 0: #only if we have bombs
		bombmap[bombs[:,0], bombs[:,1]] = bombs[:,2]
	
	explosionmap = np.array(data["explosions"])
	
	coinmap = np.zeros(arena.shape)
	coins = np.array(data["coins"])
	
	if coins.shape[0] > 0: #only if we have coins
		coinmap[coins[:,0], coins[:,1]] = np.ones(coins.shape[0])
	
	#put it together into a vector
	vec = np.concatenate([arena.ravel(), bombmap.ravel(), explosionmap.ravel(), coinmap.ravel(), player_pos, player_canbomb])
	
	return torch.from_numpy(np.array(vec)).float()  #[vec]


class ReplayBuffer(Dataset):
	def __init__(self, dev, maxsize):
		super(ReplayBuffer, self).__init__()
		self.experiences = []
		self.maxsize = maxsize
		self.device = dev
	
	def add(self, state, action, reward, nextstate, terminal):
		if len(self.experiences) == self.maxsize:
			self.experiences.pop(0) #remove oldest element
		self.experiences.append([state, nextstate, torch.tensor([action]), torch.FloatTensor([reward]), terminal])
	
	def clear(self):
		self.experiences = []
	
	def __len__(self):
		return len(self.experiences)
	
	def __getitem__(self, i):
		if i > (len(self.experiences)-1):
			print("Problem! i={}, size={}".format(i, len(self.experiences)))
			return 0
		state = self.experiences[i][0]
		nextstate = self.experiences[i][1]
		action = self.experiences[i][2]
		reward = self.experiences[i][3]
		terminal = self.experiences[i][4]
		
		if not state.is_cuda:
			state = state.to(self.device)
		if not nextstate.is_cuda:
			nextstate = nextstate.to(self.device)
		if not action.is_cuda:
			action = action.to(self.device)
		if not reward.is_cuda:
			reward = reward.to(self.device)
		
		return {"state" : state, "nextstate" : nextstate, "action" : action, "reward" : reward, "terminal" : terminal}
