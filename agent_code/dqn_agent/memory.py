"""
This file contains code for the experience replay buffer
"""
import random
import numpy as np
import torch

from settings import s

def pack_scene_old(data):
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
	vec = np.concatenate([arena.ravel(), bombmap.ravel(), explosionmap.ravel(), coinmap.ravel(), player_pos, player_canbomb, opponents_pos])
	
	return torch.from_numpy(np.array(vec)).float()  #[vec]


def pack_scene(data):
	"""
	Collect data from current game_state and format data
	into one big vector which then can be used to plan actions
	"""
	#arena map
	arena = np.array(data["arena"])
	explosionmap = np.array(data["explosions"])
	
	player_pos = np.array(data["self"][0:2])
	player_canbomb = np.array([data["self"][3]])
	#playerinfo = np.array(data["self"])
	
	opponents_info = np.zeros((s.max_agents, 3))
	others_pos = np.array(data["others"])[:,0:2]
	others_canbomb = np.array(data["others"])[:,3]
	if others_pos.shape[0] > 0:
		opponents_info[0:others_pos.shape[0], 0:2] = others_pos
		opponents_info[0:others_pos.shape[0], 2] = others_canbomb
	else:
		opponents_info.fill(-1)
	
	bombinfo = np.zeros((s.max_agents, 3))
	bombs = np.array(data["bombs"])
	if bombs.shape[0] > 0:
		bombinfo[0:bombs.shape[0]] = bombs
	else:
		bombinfo.fill(-1)
	
	coininfo = np.zeros((9,2))
	coins = np.array(data["coins"]) #9*2
	if coins.shape[0] > 0:
		coins[0:coins.shape[0]] = coins
	else:
		coininfo.fill(-1)
	
	#put it together into a vector
	vec = np.concatenate([arena.ravel(), explosionmap.ravel(), player_pos, player_canbomb, bombinfo.ravel(), coininfo.ravel(), opponents_info.ravel()])
	
	return torch.from_numpy(np.array(vec)).float()  #[vec]


#####################################
class PrioritizedReplayBuffer(object):
	def __init__(self, dev, capacity):
		super(PrioritizedReplayBuffer, self).__init__()
		self.capacity = capacity
		self.experiences = [] #array containing the experiences (state, action, reward, nextstate)-pairs
		self.pointer = 0 #index where to put the new error
		self.data = np.zeros(2*self.capacity-1) #node and leave data -> use Kekule numbers to access
		self.device = dev
		
		self.max_priority = 1.0 #clipped priorities
		self.param_a = 0.6 #paper
		self.param_b = 0.4
		self.bincrease= 0.001
		self.epsilon = 0.01
	
	def __len__(self):
		return len(self.experiences)
		
	
	def add(self, state, action, reward, nextstate, terminal):
		if len(self.experiences) == self.capacity: #if buffer is full, replace
			self.experiences[self.pointer] = [state.unsqueeze(0), nextstate.unsqueeze(0), torch.tensor([action]), torch.FloatTensor([reward]), torch.ByteTensor([terminal])]
		else: #insert new one
			self.experiences.append([state.unsqueeze(0), nextstate.unsqueeze(0), torch.tensor([action]), torch.FloatTensor([reward]), torch.ByteTensor([terminal])])
		
		init_val = np.amax(self.data[self.capacity-1:]) #maximum priority
		if init_val == 0:
			init_val = self.max_priority
		
		self.data[(self.pointer+self.capacity-1)] = 0 #initiate with small error such that new event is revisited
		self.update(self.pointer, init_val) #update with init_val
		self.pointer += 1
		if self.pointer >= self.capacity:
			self.pointer = 0 #start from the beginning
	
	def update(self, index, value): #is used after add
		parent_index = index+self.capacity-1
		diff = value - self.data[parent_index]
		self.data[parent_index] += diff #update current leaf
		while parent_index > 0:
			parent_index = (parent_index-1)//2
			self.data[parent_index] += diff #update parent node
	
	def update_many(self, indices, values): #update from training
		newvals = np.minimum(values+self.epsilon, self.max_priority)
		for i in range(indices.shape[0]):
			self.update(indices[i], newvals[i])
	
	def walk(self, value):
		index = 0
		while index<(2*self.capacity-1):
			#get childs
			left = 2*index+1
			right = left + 1
			
			if left>(2*self.capacity-2):
				break
			
			if self.data[left] >= value:
				index = left
			else:
				value -= self.data[left]
				index = right
		
		return index-self.capacity+1
	
	def walk_many(self, values):
		indices = np.zeros(values.shape, dtype=np.int32)
		while indices[0] <(2*self.capacity-1):
			#get childs
			left = 2*indices+1
			right = left+1
			
			if left[0]>(2*self.capacity-2):
				break
			
			vals = self.data[left]
			i = np.where(vals>=values)[0]
			indices[i] = left[i]
			i = np.where(vals<values)[0]
			indices[i] = right[i]
			values[i] -= vals[i]
		
		return indices
		
	def sample(self, batch_size, use_priority=True):
		if not use_priority:
			batch = random.sample(self.experiences, batch_size)
			batch = np.array(batch, dtype=object)
			states = torch.cat(batch[:,0].tolist()).to(self.device)
			nextstates = torch.cat(batch[:,1].tolist()).to(self.device)
			actions = torch.cat(batch[:,2].tolist()).to(self.device)
			rewards = torch.cat(batch[:,3].tolist()).to(self.device)
			terminals = torch.cat(batch[:,4].tolist()).to(self.device)
			
			return states, nextstates, actions, rewards, terminals, 1.0, 0
		else:
			#increase b
			self.param_b = np.minimum(1.0, self.param_b+self.bincrease)
			#divide range into subranges
			segment_size = self.data[0]/float(batch_size)
			
			lower = np.arange(batch_size)*segment_size
			upper = (np.arange(batch_size)+1.)*segment_size
			drawn = np.random.uniform(lower, upper) # uniformly draw inbetween ranges
			
			indices = np.array([self.walk(x) for x in drawn])
			#indices = self.walk_many(drawn)
			batch = [self.experiences[x] for x in indices]
			batch = np.array(batch, dtype=object)
			states = torch.cat(batch[:,0].tolist()).to(self.device)
			nextstates = torch.cat(batch[:,1].tolist()).to(self.device)
			actions = torch.cat(batch[:,2].tolist()).to(self.device)
			rewards = torch.cat(batch[:,3].tolist()).to(self.device)
			terminals = torch.cat(batch[:,4].tolist()).to(self.device)
			
			#calculate importance sampling weights
			ind = indices + (self.capacity-1)
			priorities = self.data[ind]
			probabilities = np.power(priorities, self.param_a)/np.sum(np.power(self.data[self.capacity-1:], self.param_a))
			
			sampling_weights = np.power(probabilities*len(self.experiences), -self.param_b)
			sampling_weights /= np.amax(sampling_weights)
			sampling_weights = torch.DoubleTensor(sampling_weights).to(self.device)
			
			return states, nextstates, actions, rewards, terminals, sampling_weights, indices
			
