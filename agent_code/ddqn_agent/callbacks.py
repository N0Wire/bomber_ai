import numpy as np
from time import time, sleep
import pygame

from settings import s
from settings import e

#user stuff
import torch
import torch.nn as nn
import torch.optim as optim #for ADAM optimizer
from torch.utils.data import DataLoader
from .model import FFQN
from .memory import pack_scene, ReplayBuffer
from .rewards import rewards_normal, rewards_clipped
from .exp import greedy_epsilon

import os
import matplotlib.pyplot as plt


#TODO
#- use multiple frames as state (at least 2 vectors)


#######################
#Variables
RUN_NAME = "Run_1"
CONTINUE_TRAIN = False
DEVICE = 0

EPSILON_DECAY = 0.0001
EPSILON_FINAL = 0.1

TARGET_UPDATE = 200
BATCH_SIZE = 32
EXPERIENCEBUF_SIZE = 10000 #number of experiences stored in experience replay buffer
GAMMA = 0.95
MIN_EXPERIENCES = 200




def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.xavier_normal_(m.weight)
		#m.bias.data.fill_(0.01)


def setup(self):
	global RUN_NAME
	global DEVICE
	global EXPERIENCEBUF_SIZE
	global CONTINUE_TRAIN

	start = time()
	#initialize CUDA
	dev = "cuda:{}".format(DEVICE) if torch.cuda.is_available() else "cpu"
	self.device = torch.device(dev)
	self.logger.info("Using device: " + dev)
	
	
	self.memory = ReplayBuffer(self.device, EXPERIENCEBUF_SIZE)
	self.oldstate = 0
	self.lastaction = s.actions[5] #default = 'WAIT'
	self.epsilon = 1.0
	self.numepisodes = 0
	self.totalsteps = 0
	
	self.step_rewards = [] #rewards per episode
	self.avg_rewards = [] #average rewards per episode
	self.step_losses = []
	self.avg_losses = []
	self.current_steps = 0
	self.steps_per_episode = []
	
	
	#initialize model, optimizer and loss
	#calculate input length
	inlen = 4*s.rows*s.cols+3
	
	self.pnet = FFQN(inlen, len(s.actions)) #polciy net
	self.pnet.to(self.device) #we only use device 0
	
	#try to load latest model
	curpath = os.path.dirname(os.path.realpath(__file__))
	path = os.path.join(curpath, "checkpoints")
	path = os.path.join(path, RUN_NAME)
	if not os.path.isdir(path):
		os.makedirs(path)
	
	files = [f for f in os.listdir(path) if f.find(".pth") > 0]
	files.sort()
	if len(files) > 0 and CONTINUE_TRAIN:
		self.pnet.load_state_dict(torch.load(path+"/"+files[-1]))
		pos = files[-1].find(".pth")
		if pos > 0:
			self.numepisodes = int(files[-1][8:pos]) #for continuing training
		
		self.avg_rewards = np.load(path+"/rewards.npy").tolist()
		self.avg_losses = np.load(path+"/losses.npy").tolist()
		self.steps_per_episode = no.load(path+"/steps.npy").tolist()
	else:
		self.pnet.apply(init_weights)
	
	self.tnet = FFQN(inlen, len(s.actions)) #target net - only updated every C times
	self.tnet.to(self.device)
	self.tnet.load_state_dict(self.pnet.state_dict()) #initial target net is same as policy net
	
	#choose ADAM optimizer with standard settings
	self.optimizer = optim.Adam(self.pnet.parameters(), lr=0.001)
	
	#initialize loss - L1
	self.loss = nn.SmoothL1Loss() #maybe choose Huber loss here
	
	self.logger.info("Client initialized in {:.2f}s".format(time()-start))


def act(self):
	global EPSILON_DECAY
	global EPSILON_FINAL
	
	self.nextaction = s.actions[5] #default is wait
	self.current_steps += 1
	
	#access scene information and build data from it
	self.currentstate = pack_scene(self.game_state)
	cstate = self.currentstate.to(self.device)
	
	if self.game_state["train"]:
		#do exploration/exploitation to choose action
		if self.epsilon > 0.1:
			self.epsilon -= EPSILON_DECAY
		self.lastaction = greedy_epsilon(self.pnet, cstate, self.epsilon) #most of world is known, therefore exploit more
		self.lastaction = self.lastaction
		self.next_action = s.actions[self.lastaction]
	else:
		#just do a single forward pass
		with torch.no_grad():
			self.next_action = s.actions[self.pnet(cstate).max(0)[1]]


def reward_update(self):
	#collect new experience and store it
	newstate = pack_scene(self.game_state)
	
	#get reward
	totalreward = 0
	is_terminal = 0
	for ev in self.events:
		totalreward += rewards_clipped[ev] #rewards_clipped[e]
		if ev == e.KILLED_SELF or ev == e.GOT_KILLED or ev == e.SURVIVED_ROUND:
			is_terminal = 1
	
	self.step_rewards.append(totalreward)
	self.memory.add(self.currentstate, self.lastaction, totalreward, newstate, is_terminal)


def end_of_episode(self):
	global RUN_NAME
	global BATCH_SIZE
	global TARGET_UPDATE
	global GAMMA
	global MIN_EXPERIENCES
	
	start = time()
	
	#look for final state
	newstate = pack_scene(self.game_state)
	totalreward = 0
	is_terminal = 0
	for ev in self.events:
		totalreward += rewards_clipped[ev]
		if ev == e.KILLED_SELF or ev == e.GOT_KILLED or ev == e.SURVIVED_ROUND:
			is_terminal = 1
			break #always got killed_self and got_killed one after each other
	
	self.step_rewards.append(totalreward)
	self.memory.add(self.currentstate, self.lastaction, totalreward, newstate, is_terminal)
	
	if len(self.memory) >= MIN_EXPERIENCES:
		#sample experiences from buffer with batch_size
		loader = DataLoader(self.memory, BATCH_SIZE, shuffle=True)
		
		for i, batch in enumerate(loader):
			cstates = batch["state"]
			nstates = batch["nextstate"]
			actions = batch["action"]
			rewards = batch["reward"]
			terminals = batch["terminal"]
			
			
			#calculate q values and loss
			qs = self.pnet(cstates).gather(1, actions) #actual q values -> only select those which belong to the taken actions
			nextq = 0
			with torch.no_grad():
				nextq = self.tnet(nstates)[self.pnet(nstates).max(1)[0].long()] #let the policy net decide the action
			#nextq = self.tnet(nstates).max(1)[0].detach() #no backprop needed here
			for t in range(terminals.size()[0]):
				if terminals[t] == 1:
					nextq[t] = 0
			nextq = nextq*GAMMA+rewards
			
			loss = self.loss(qs, nextq)
			
			#do back propagation
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			#store values for plotting
			self.step_losses.append(loss.item())
			
			#update target net
			self.totalsteps += 1
			if self.totalsteps % TARGET_UPDATE == 0:
				self.tnet.load_state_dict(self.pnet.state_dict())
		
	
	curpath = os.path.dirname(os.path.realpath(__file__))
	self.numepisodes += 1
	
	#visualization
	if self.numepisodes % 10 == 0:
		self.avg_rewards.append(np.mean(self.step_rewards))
		self.avg_losses.append(np.mean(self.step_losses))
		self.steps_per_episode.append(self.current_steps)
	
	if self.numepisodes % 20 == 0:
		plt.figure(1, dpi=200)
		plt.title("Average Reward per Episode")
		plt.xlabel("Episode")
		plt.ylabel("Reward")
		plt.plot(np.arange(len(self.avg_rewards))*10, self.avg_rewards)
		plt.savefig(os.path.join(curpath, "checkpoints", RUN_NAME, "rewards.pdf"))
		plt.clf()
		
		plt.figure(2, dpi=200)
		plt.title("Average Loss per Episode")
		plt.xlabel("Episode")
		plt.ylabel("Loss")
		plt.plot(np.arange(len(self.avg_losses))*10, self.avg_losses)
		plt.savefig(os.path.join(curpath, "checkpoints", RUN_NAME, "losses.pdf"))
		plt.clf()
		
		plt.figure(3, dpi=200)
		plt.title("Steps per Episode")
		plt.xlabel("Episode")
		plt.ylabel("Steps")
		plt.plot(np.arange(len(self.steps_per_episode))*10, self.steps_per_episode)
		plt.savefig(os.path.join(curpath, "checkpoints", RUN_NAME, "steps.pdf"))
		plt.clf()
	
	self.step_rewards = []
	self.step_losses = []
	self.current_steps = 0
	
	if self.numepisodes % 100 == 0: #save every 100 episode
		#save model
		path = os.path.join(curpath, "checkpoints", RUN_NAME)
		path = os.path.join(path, "episode_{}.pth".format(self.numepisodes))
		torch.save(self.pnet.state_dict(), path)
		np.save(os.path.join(curpath, "checkpoints", RUN_NAME, "rewards.npy"), np.array(self.avg_rewards))
		np.save(os.path.join(curpath, "checkpoints", RUN_NAME, "losses.npy"), np.array(self.avg_losses))
		np.save(os.path.join(curpath, "checkpoints", RUN_NAME, "steps.npy"), np.array(self.steps_per_episode))
	
	self.logger.info("Training Episode {} took {:.2f}s".format(self.numepisodes, time()-start))
