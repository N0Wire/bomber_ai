"""
This file contains code for the experience replay buffer
"""
import random
import numpy as np
import torch

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


#####################################
class PrioritizedReplayBuffer(object):
	def __init__(self, dev, capacity):
		super(PrioritizedReplayBuffer, self).__init__()
		self.capacity = capacity
		self.experiences = [] #array containing the experiences (state, action, reward, nextstate)-pairs
		self.pointer = 0 #index where to put the new error
		self.data = np.zeros(2*self.capacity-1) #node and leave data -> use Kekule numbers to access
		self.device = dev
		
		self.max_priority = 1. #clipped priorities
		self.param_a = 0.6 #paper
		self.param_b = 0.4
		self.bincrease= 0.001
		self.epsilon = 0.01
	
	def __len__(self):
		return len(self.experiences)
	
	def propagate(self, index, change):
		self.data[index] += change #update node
		#get parent
		parent_index = (index-1)//2
		if parent_index > 0: #we are not at the root node
			self.propagate(parent_index, change)
		
	
	def add(self, state, action, reward, nextstate, terminal):
		global INITIAL_PRIORITY
		if len(self.experiences) == self.capacity: #if buffer is full, replace
			self.experiences[self.pointer] = [state.unsqueeze(0), nextstate.unsqueeze(0), torch.tensor([action]), torch.FloatTensor([reward]), torch.ByteTensor([terminal])]
		else: #insert new one
			self.experiences.append([state.unsqueeze(0), nextstate.unsqueeze(0), torch.tensor([action]), torch.FloatTensor([reward]), torch.ByteTensor([terminal])])
		
		init_val = np.amax(self.data[self.capacity-1:]) #maximum priority
		if init_val == 0:
			init_val = self.max_priority
		
		self.data[self.pointer+self.capacity-1] = init_val #initiate with small error such that new event is revisited
		self.pointer += 1
		if self.pointer >= self.capacity:
			self.pointer = 0 #start from the beginning
	
	def update(self, index, value):
		tree_index = self.capacity-1+index
		diff = value - self.data[tree_index]
		self.data[tree_index] = value #update leaf
		self.propagate(tree_index, diff)
	
	def update_many(self, indices, values):
		tree_indices = indices+self.capacity-1
		newvals = np.min(values+self.epsilon, self.max_priority)
		diffs = newvals - self.data[tree_indices]
		for i in range(diffs.shape[0]):
			self.propagate(tree_indices[i], diffs[i])
	
	def walk(self, value, index=0):
		#get childs
		left = 2*index+1
		right = left + 1
		
		#check if we arrived at a leaf
		if left > (2*self.capacity-2):
			return index-self.capacity-1 #index in experience-array
		
		if self.data[left] >= value:
			return self.walk(value, left)
		else:
			return self.walk(value-self.data[left], right)
	
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
			self.param_b = np.min([1.0, self.param_b+self.bincrease])
			#divide range into subranges
			segment_size = self.data[0]/batch_size
			
			lower = np.arange(batch_size)*segment_size
			upper = (np.arange(batch_size)+1)*segment_size
			drawn = np.random.uniform(lower, upper) # uniformly draw inbetween ranges
			
			indices = np.array([self.walk(x) for x in drawn])
			print(indices)
			batch = [self.experiences[x] for x in indices]
			batch = np.array(batch, dtype=object)
			states = torch.cat(batch[:,0].tolist()).to(self.device)
			nextstates = torch.cat(batch[:,1].tolist()).to(self.device)
			actions = torch.cat(batch[:,2].tolist()).to(self.device)
			rewards = torch.cat(batch[:,3].tolist()).to(self.device)
			terminals = torch.cat(batch[:,4].tolist()).to(self.device)
			
			#calculate importance sampling weights
			priorities = self.data[indices+(self.capacity-1)]
			probabilities = priorities**self.param_a/np.sum(self.data[self.capacity-1:]**self.param_a)
			
			sampling_weights = np.power(1.0/len(self.experiences)/probabilities, self.param_b)
			sampling_weights /= np.amax(sampling_weights)
			
			return states, nextstates, actions, rewards, terminals, sampling_weights, indices
			
