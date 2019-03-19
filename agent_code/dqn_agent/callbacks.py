import numpy as np
from time import time

from settings import s
from settings import e

#user stuff
import torch
import torch.nn as nn
import torch.optim as optim #for ADAM optimizer
from .model import FFQN, DuelingFFQN
from .memory import pack_scene, PrioritizedReplayBuffer
from .rewards import rewards_default, rewards_clipped
from .exp import greedy_epsilon

import os
import matplotlib.pyplot as plt


#######################
#Variables
RUN_NAME = "test"
CONTINUE_TRAIN = False
DEVICE = 0

EPSILON_DECAY = 0.0001
EPSILON_FINAL = 0.1

TARGET_UPDATE = 400 #200
BATCH_SIZE = 128 #32
EXPERIENCEBUF_SIZE = 10000 #number of experiences stored in experience replay buffer
GAMMA = 0.95
MIN_EXPERIENCES = 200
EXPERIENCE_SIZE = 2

USE_NET = "dueling" #"default"
USE_DOUBLE = True
USE_PER = False

def init_weights(m):
	if type(m) == nn.Linear:
		nn.init.xavier_normal_(m.weight)
		m.bias.data.fill_(0.01)


def setup(self):
	global RUN_NAME
	global DEVICE
	global EXPERIENCEBUF_SIZE
	global CONTINUE_TRAIN
	global EXPERIENCE_SIZE
	global USE_NET

	start = time()
	#initialize CUDA
	dev = "cuda:{}".format(DEVICE) if torch.cuda.is_available() else "cpu"
	self.device = torch.device(dev)
	self.logger.info("Using device: " + dev)
	
	
	self.memory = PrioritizedReplayBuffer(self.device, EXPERIENCEBUF_SIZE)
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
	#inlen = 4*s.rows*s.cols+3
	#explosion map+arena + self + coins + bombs + opponents
	inlen = (2*s.rows*s.cols+3+9*2+s.max_agents*3+s.max_agents*2)*EXPERIENCE_SIZE
	
	self.pnet = 0 #policy net
	if USE_NET == "dueling":
		self.pnet = DuelingFFQN(inlen, len(s.actions))
	else:
		self.pnet = FFQN(inlen, len(s.actions))
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
		self.steps_per_episode = np.load(path+"/steps.npy").tolist()
	else:
		self.pnet.apply(init_weights)
	
	self.tnet = 0 #target net - only updated every C times
	if USE_NET == "dueling":
		self.tnet = DuelingFFQN(inlen, len(s.actions))
	else:
		self.tnet = FFQN(inlen, len(s.actions))
	self.tnet.to(self.device)
	self.tnet.load_state_dict(self.pnet.state_dict()) #initial target net is same as policy net
	
	#choose ADAM optimizer with standard settings
	self.optimizer = optim.Adam(self.pnet.parameters(), lr=0.001)
	
	#initialize loss - L1
	self.loss = nn.SmoothL1Loss(reduction="none") #maybe choose Huber loss here
	
	#state buffer
	self.statebuffer = []
	
	self.logger.info("Client initialized in {:.2f}s".format(time()-start))


def act(self):
	global EPSILON_DECAY
	global EPSILON_FINAL
	global EXPERIENCE_SIZE
	
	self.nextaction = "WAIT" #default is wait
	self.current_steps += 1
	
	#at the beginning of each episode we need to fill the state buffer with the same values (no first round is wasted)
	if self.current_steps == 1:
		self.statebuffer = []
		state = pack_scene(self.game_state)
		for i in range(EXPERIENCE_SIZE):
			self.statebuffer.append(state)
	
	#build current world state
	self.currentstate = torch.cat(self.statebuffer)
	cstate = self.currentstate.to(self.device)
	
	if self.game_state["train"]:
		#do exploration/exploitation to choose action
		if self.epsilon > 0.1:
			self.epsilon -= EPSILON_DECAY
		self.lastaction = greedy_epsilon(self.pnet, cstate, self.epsilon) #other options would could be tried here
		self.next_action = s.actions[self.lastaction] #get action string
	else:
		#just do a single forward pass
		with torch.no_grad():
			self.next_action = s.actions[self.pnet(cstate).max(0)[1]]


def train_model(memory, pnet, tnet, loss, optimizer, totalsteps, step_losses):
	global BATCH_SIZE
	global USE_DOUBLE
	global GAMMA
	global USE_PER
	
	if len(memory) >= MIN_EXPERIENCES:
		#sample experiences from buffer with batch_size
		
		cstates, nstates, actions, rewards, terminals, weights, indices = memory.sample(BATCH_SIZE, use_priority=USE_PER)
		
		#calculate q values and loss
		qs = pnet(cstates).gather(1, actions.unsqueeze(1)).squeeze() #actual q values -> only select those which belong to the taken actions
		
		nextq = 0
		with torch.no_grad():
			if USE_DOUBLE:
				indices = pnet(nstates).max(1)[1].long()
				nextq = tnet(nstates).gather(1, indices.unsqueeze(1)).squeeze()
			else:
				nextq = tnet(nstates).max(1)[0] #no backprop needed here
			nextq[terminals] = 0
			nextq = nextq*GAMMA+rewards
		
		error = loss(qs, nextq).double() * weights
		error = error.mean()
		
		#do back propagation
		optimizer.zero_grad()
		error.backward()
		optimizer.step()
		
		#store values for plotting
		step_losses.append(error.item())
		
		if USE_PER:
			errors = torch.abs(qs-nextq).cpu().detach().numpy()
			#update priority
			memory.update_many(indices, errors)
		#update target net
		totalsteps += 1
		if totalsteps % TARGET_UPDATE == 0:
			tnet.load_state_dict(pnet.state_dict())
		
	return totalsteps


def reward_update(self):	
	#collect new experience and store it
	state = pack_scene(self.game_state)
	self.statebuffer.pop(0)
	self.statebuffer.append(state)
	newstate = torch.cat(self.statebuffer)
	
	#get reward
	totalreward = 0
	is_terminal = 0
	for ev in self.events:
		totalreward += rewards_default[ev] #rewards_clipped[e]
		if ev == e.KILLED_SELF or ev == e.GOT_KILLED or ev == e.SURVIVED_ROUND:
			is_terminal = 1
	
	self.step_rewards.append(totalreward)
	self.memory.add(self.currentstate, self.lastaction, totalreward, newstate, is_terminal)
	
	self.totalsteps = train_model(self.memory, self.pnet, self.tnet, self.loss, self.optimizer, self.totalsteps, self.step_losses)


def end_of_episode(self):
	global RUN_NAME
	global BATCH_SIZE
	global TARGET_UPDATE
	global GAMMA
	global MIN_EXPERIENCES
	global USE_DOUBLE
	
	if self.current_steps <= (EXPERIENCE_SIZE-1): #ignore first rounds
		return
	
	#collect new experience and store it
	state = pack_scene(self.game_state)
	self.statebuffer.pop(0)
	self.statebuffer.append(state)
	newstate = torch.cat(self.statebuffer)
	
	#look for final state
	totalreward = 0
	is_terminal = False
	for ev in self.events:
		totalreward += rewards_default[ev]
		if ev == e.KILLED_SELF or ev == e.GOT_KILLED or ev == e.SURVIVED_ROUND:
			is_terminal = True
			break #always got killed_self and got_killed one after each other
	
	self.step_rewards.append(totalreward)
	self.memory.add(self.currentstate, self.lastaction, totalreward, newstate, is_terminal)
	
	self.totalsteps = train_model(self.memory, self.pnet, self.tnet, self.loss, self.optimizer, self.totalsteps, self.step_losses)		
	
	curpath = os.path.dirname(os.path.realpath(__file__))
	self.numepisodes += 1
	
	#visualization
	if self.numepisodes % 50 == 0:
		self.avg_rewards.append(np.mean(self.step_rewards))
		self.avg_losses.append(np.mean(self.step_losses))
		self.steps_per_episode.append(self.current_steps)
	
	if self.numepisodes % 50 == 0:
		plt.figure(1, dpi=200)
		plt.title("Average Reward per 50 Episodes")
		plt.xlabel("Episode")
		plt.ylabel("Reward")
		plt.plot(np.arange(len(self.avg_rewards))*50, self.avg_rewards)
		plt.savefig(os.path.join(curpath, "checkpoints", RUN_NAME, "rewards.pdf"))
		plt.clf()
		
		plt.figure(2, dpi=200)
		plt.title("Average Loss per 50 Episodes")
		plt.xlabel("Episode")
		plt.ylabel("Loss")
		plt.plot(np.arange(len(self.avg_losses))*50, self.avg_losses)
		plt.savefig(os.path.join(curpath, "checkpoints", RUN_NAME, "losses.pdf"))
		plt.clf()
		
		plt.figure(3, dpi=200)
		plt.title("Average Steps per 50 Episodes")
		plt.xlabel("Episode")
		plt.ylabel("Steps")
		plt.plot(np.arange(len(self.steps_per_episode))*50, self.steps_per_episode)
		plt.savefig(os.path.join(curpath, "checkpoints", RUN_NAME, "steps.pdf"))
		plt.clf()
	
	self.step_rewards = []
	self.step_losses = []
	self.current_steps = 0
	
	if self.numepisodes % 200 == 0: #save every 100 episode
		#save model
		path = os.path.join(curpath, "checkpoints", RUN_NAME)
		path = os.path.join(path, "episode_{}.pth".format(self.numepisodes))
		torch.save(self.pnet.state_dict(), path)
		np.save(os.path.join(curpath, "checkpoints", RUN_NAME, "rewards.npy"), np.array(self.avg_rewards))
		np.save(os.path.join(curpath, "checkpoints", RUN_NAME, "losses.npy"), np.array(self.avg_losses))
		np.save(os.path.join(curpath, "checkpoints", RUN_NAME, "steps.npy"), np.array(self.steps_per_episode))
