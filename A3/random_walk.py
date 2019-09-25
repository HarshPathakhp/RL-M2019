import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
(R = 0)X - A - B - C - D - E - X (R = 1) 
 "-" implies bidirectional edge with reward 0
"""
class MC:
	def __init__(self, alpha, episodes):
		self.alpha = alpha
		self.rwalk = RandomWalk()
		self.episodes = episodes
	def task2(self, runs = 100):
		true_estimate = np.zeros((runs,self.episodes+1, 7))
		for i in range(true_estimate.shape[0]):
			for j in range(true_estimate.shape[1]):
				true_estimate[i,j] = np.array([0,1/6,2/6,3/6,4/6,5/6,0])
		run_out = []
		for idx in range(runs):
			X = np.ones((7))
			X.fill(0.5)
			X[0] = 0
			X[6] = 0
			value = X.copy()
			for eps in range(self.episodes):
				seq = []
				cur_state = 3
				while(True):
					next_state, action, reward = self.rwalk.get_next_state(cur_state)
					seq.append(([cur_state, action], reward))
					cur_state = next_state
					if(next_state == 0 or next_state == 6):
						break
				for idx in range(len(seq)):
					state = seq[idx][0][0]
					action = seq[idx][0][1]
					returns = seq[len(seq) - 1][1] #since gamma is 1 and rest awards are 0
					value[state] += self.alpha * (returns - value[state])
				X = np.vstack([X,value])
			run_out.append(X)
		x = np.array(run_out)
		error = x - true_estimate
		rms_error = np.linalg.norm(error, axis = 2)
		rms_error = np.mean(rms_error, axis = 0)
		return rms_error
class TemporalDifference:
	def __init__(self, alpha, episodes):
		self.alpha = alpha
		self.rwalk = RandomWalk()
		self.episodes = episodes
		self.values = np.ones((7))
		self.values.fill(0.5)
		self.values[0] = 0
		self.values[6] = 0
		self.plot_checkpoints = [0,1,10,100]
		self.colors = ['black', 'red', 'green', 'blue']
	def task1(self):
		""" updates v(s) for the rwalk object using the behavior policy in rwalk object"""
		for episode in tqdm(range(self.episodes)):
			for idx in range(len(self.plot_checkpoints)):
				if(episode == self.plot_checkpoints[idx]):
					plt.plot(['A', 'B', 'C', 'D', 'E'], self.values[1:-1], color = self.colors[idx], label = "episode - " + str(episode))
			cur_state = self.rwalk.initial_state
			while(True):
				next_state, action, reward = self.rwalk.get_next_state(cur_state)
				self.values[cur_state] += self.alpha * (reward + self.values[next_state] - self.values[cur_state])
				if(next_state == 0 or next_state == 6):
					break
				cur_state = next_state
		plt.plot(['A', 'B', 'C', 'D', 'E'], [i/6 for i in range(1,6)], color = "orange", label = "truth")
		plt.legend()
		plt.savefig("./fig5_2_left.png")
		plt.close()
		return self.values	
	
	def task2_temporal(self, runs = 100):
		true_estimate = np.zeros((runs,self.episodes+1, 7))
		for i in range(true_estimate.shape[0]):
			for j in range(true_estimate.shape[1]):
				true_estimate[i,j] = np.array([0,1/6,2/6,3/6,4/6,5/6,0])
		run_out = []
		for idx in range(runs):
			X = np.ones((7))
			X.fill(0.5)
			X[0] = 0
			X[6] = 0
			for eps in range(self.episodes):
				cur_state = self.rwalk.initial_state
				v2 = self.values.copy()
				while(True):
					next_state, action, reward = self.rwalk.get_next_state(cur_state)
					v2[cur_state] += self.alpha * (reward + v2[next_state] - v2[cur_state])
					if(next_state == 0 or next_state == 6):
						break
					cur_state = next_state
				X = np.vstack([X, v2])
				self.values = v2
			run_out.append(X)
		x = np.array(run_out)
		error = x - true_estimate
		rms_error = np.linalg.norm(error, axis = 2)
		rms_error = np.mean(rms_error, axis = 0)
		return rms_error
class RandomWalk():
	def __init__(self):
		self.ACTION_LEFT = -1
		self.ACTION_RIGHT = 1
		self.initial_state = 3 #state C		
	def behavior_policy(self, state):
		""" returns the action to take for a given state, to generate episodes
			states are 1 indexed, i.e, state A starts from 1
		"""
		toss = np.random.binomial(1, 0.5)
		return self.ACTION_LEFT if toss == 1 else self.ACTION_RIGHT
	
	def get_next_state(self, state):
		if(state == 0 or state == 6): #terminal states
			return -1			
		next_action = self.behavior_policy(state)
		next_state = state + next_action
		reward = 0
		if(next_state == 6):
			reward = 1
		self.state_now = next_state
		return next_state, next_action, reward

alpha_td = [0.15,0.1,0.05]
col = ['black', 'red', 'green', 'blue']
alpha_mc = [0.01,0.02,0.03,0.04]

episodes = [i for i in range(0,101)]
for j,i in enumerate(alpha_td):
	td = TemporalDifference(i, int(100))
	error = td.task2_temporal()
	plt.plot(episodes, error, color = col[j], label = 'td a = ' + str(i))
for j,i in enumerate(alpha_mc):
	mc = MC(i, int(100))
	error = mc.task2()
	plt.plot(episodes, error, color = col[j], label = 'mc a = ' + str(i), linestyle = "dashed")
plt.legend()
plt.savefig("./fig5_2_right.png")
plt.close()
