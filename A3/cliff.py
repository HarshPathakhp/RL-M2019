"""
x x x x x x x x x x x x
x x x x x x x x x x x x
x x x x x x x x x x x x
S c c c c c c c c c c G
Reward is -100 for jumping into cliff
Reward is -1 for any other transition
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
def SARSA(episodes, alpha = 0.1, runs = 100):
	ret = []
	for run in range(runs):
		print(run)
		sor = []
		gw = Gridworld(e = 0.2)
		for eps in range(episodes):
			cur_state = gw.START
			sum_of_rewards = 0
			counter = 0
			action = gw.behavior_policy(cur_state)
			while(cur_state != gw.GOAL):
				reward, next_state = gw.take_action(cur_state, action)
				sum_of_rewards += reward
				if(next_state == gw.GOAL):
					gw.set_Q(cur_state, action, (1-alpha)*(gw.get_Q(cur_state, action)) + (alpha)*(reward))
					cur_state = next_state
					continue
				action_prime = gw.behavior_policy(next_state)
				if(next_state in gw.cliff_cells):
					print("oops")
				gw.set_Q(cur_state, action, (1-alpha)*(gw.get_Q(cur_state, action)) + (alpha)*(reward + gw.get_Q(next_state, action_prime)))
				cur_state = next_state
				action = action_prime
			sor.append(sum_of_rewards)
		ret.append(sor)
	ret = np.array(ret)
	return np.mean(ret, axis = 0)
def Qlearning(epsiodes, alpha = 0.1, runs = 100):
	ret = []
	for run in range(runs):
		print(run)
		sor = []
		gw = Gridworld(e = 0.2)
		for eps in range(episodes):
			cur_state = gw.START 
			sum_of_rewards = 0
			counter = 0
			while(cur_state != gw.GOAL):
				counter += 1
				#print(cur_state, end = " ")
				if(cur_state in gw.cliff_cells):
					print("oops")
				action = gw.behavior_policy(cur_state)
				reward, next_state = gw.take_action(cur_state, action)
				sum_of_rewards += reward
				gw.set_Q(cur_state, action, gw.get_Q(cur_state, action)*(1-alpha) + alpha*(reward + gw.max_selection(next_state)))
				cur_state = next_state
			#print(counter, sum_of_rewards)
			sor.append(sum_of_rewards)
		ret.append(sor)
	ret = np.array(ret)
	return np.mean(ret, axis = 0)	
class Gridworld:	
	def __init__(self, e):
		self.START = (3,0)
		self.GOAL = (3,11)
		self.cliff_cells = []
		self.e = e #e-greedy
		for i in range(1, 11):
			self.cliff_cells.append((3,i))
		#row, col and max 4 directions in order up, down, left, right
		self.Qvalues = np.zeros((4,12,4))
			
	def is_valid(self, state):
		row = state[0]
		col = state[1]
		return (row >= 0 and row < 4 and col >= 0 and col < 12)
		#checks if a state is in the grid or lies outside,  i.e., invalid
	def get_neigbors(self, state):
		row = state[0]
		col = state[1]
		row_up = row-1
		row_down = row+1
		col_left = col - 1
		col_right = col + 1
		
		up = (row_up, col) 
		down = (row_down, col)
		left = (row, col_left)
		right = (row, col_right)
		choices = [up, down, left, right]
		action = [0,1,2,3]
		ret = []
		for idx,i in enumerate(choices):
			if self.is_valid(i):
				if(i in self.cliff_cells):
					ret.append([self.START, action[idx]])
				else:
					ret.append([i, action[idx]])
		return ret
		#returns all valid neighbors and their step rewards of a state in form of a list	
	def get_Q(self, state, action):
		return self.Qvalues[state[0], state[1], action]
	def set_Q(self, state, action, val):
		self.Qvalues[state[0], state[1], action] = val
	def take_action(self, state, action):
		next_state = None
		if(action == 0):
			next_state = (state[0]-1, state[1])
		elif(action == 1):
			next_state = (state[0]+1, state[1])
		elif(action == 2):
			next_state = (state[0],state[1]-1)
		else:
			next_state = (state[0], state[1]+1)
		reward = -1
		if(next_state in self.cliff_cells):
			reward = -100
			next_state = self.START
		return reward, next_state
	def behavior_policy(self, state):
		neighbors = self.get_neigbors(state)
		states = []
		objective = []
		actions = []
		#print(state, neighbors)
		for i in neighbors:
			states.append(i[0])
			actions.append(i[1])
			objective.append(self.get_Q(state, i[1]))
		objective = np.array(objective)
		argmax_state = states[np.argmax(objective)]
		argmax_action = actions[np.argmax(objective)]
		toss = numpy.random.rand()
		if(toss < self.e):
			any_state_index = np.random.choice(len(objective), 1)
			any_state_index = any_state_index[0]
			return actions[any_state_index]  
		else:
			return argmax_action
	def max_selection(self, state):
		#only for q learning
		neighbors = self.get_neigbors(state)
		actions = []
		for i in neighbors:
			actions.append(i[1])
		objective = []
		for i in actions:
			objective.append(self.get_Q(state, i))	
		ret = np.max(objective)
		return ret 	
episodes = int(500)
sor_sarsa = SARSA(episodes = episodes)
sor_q = Qlearning(epsiodes = episodes)
episodes = [i for i in range(1, episodes+1)]
plt.plot(episodes, sor_q, color = "blue", label = "qlearning")
plt.plot(episodes, sor_sarsa, color = "red", label = "sarsa")
plt.legend()
plt.savefig("./q7.png")
plt.close()