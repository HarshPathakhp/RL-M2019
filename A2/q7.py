import numpy as np
import sys
import math
import os
cache = {}
def get_poission_prob(x, lamda):
	if(cache.get((x,lamda), -1) != -1):
		return cache.get((x,lamda))
	cache[(x,lamda)] = np.exp(-lamda) * pow(lamda, x) / math.factorial(x)
	return cache[(x,lamda)]

max_cars = 20
val_state = np.zeros((max_cars+1,max_cars+1)) #no. of cars in location 1 and 2
policy = np.zeros((max_cars+1,max_cars+1)) #policy[i,j] denotes net transfer of cars, if negative, then transfer from loc 1 to loc 2 and vice-versa
#NOTE : -5 <= policy[i,j] <= 5

mean_rental_loc1 = 3
mean_rental_loc2 = 4
mean_return_loc1 = 3
mean_return_loc2 = 2

gamma = 0.9
rent_reward = 10
move_cost = 2
MAX_BOUND = 10
def displayp(policy, iter):
	policy = policy.astype(np.int)
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	x = [i for i in range(max_cars+1)]
	im = ax.imshow(policy)
	ax.set_xticks(np.arange(len(x)))
	ax.set_yticks(np.arange(len(x)))
	ax.set_xticklabels(x)
	ax.set_yticklabels(x)
	for i in range(len(x)):
	    for j in range(len(x)):
	        text = ax.text(j, i, int(policy[i, j]),
	                       ha="center", va="center", color="w")
	ax.set_title("Policy at iter %d"%(iter))
	os.makedirs("./q7/policy", exist_ok = True)
	fig.tight_layout()
	plt.savefig("./q7/policy/" + str(iter) + ".png")

def displayv(V, iter):
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	from matplotlib import cm
	fig = plt.figure(figsize=(8,6))
	ax_surf = fig.gca(projection='3d')
	ax_surf.set_position([0.1,0.15,0.7,0.7])
	X, Y = np.meshgrid(np.arange(0, max_cars+1), np.arange(0, max_cars+1))
	print(X.shape)
	print(Y.shape)
	surf = ax_surf.plot_surface(X, Y, V, cmap=cm.coolwarm,
								linewidth=0, antialiased=False)
	ax_surf.set_xticks(np.arange(0,max_cars+1,4).astype(int))
	ax_surf.set_yticks(np.arange(0,max_cars+1,4).astype(int))
	ax_surf.set_xlabel('Number of cars at location 1')
	ax_surf.set_ylabel('Number of cars at location 2')
	ax_surf.set_title('Values at different states. Iteration:'+str(iter))
	ax_color = fig.add_axes([0.85,0.25,0.03,0.5])
	cbar = fig.colorbar(surf, cax=ax_color, 
						orientation='vertical')
	os.makedirs("./q7/value/", exist_ok=True)
	plt.savefig("./q7/value/" + str(iter)+".png")
def expected_return(V, action, ij):
	cars_loc1 = ij[0]
	cars_loc2 = ij[1]
	returns = 0
	for rental_loc1 in range(MAX_BOUND+1):
		for rental_loc2 in range(MAX_BOUND+1):
			total_rentals_1 = min(cars_loc1, rental_loc1)
			total_rentals_2 = min(cars_loc2, rental_loc2)
			prob_rental_1 = get_poission_prob(rental_loc1, mean_rental_loc1)
			prob_rental_2 = get_poission_prob(rental_loc2, mean_rental_loc2)
			net_profit = rent_reward * (total_rentals_2 + total_rentals_1)
			movement_cost = 2 * (abs(action))
			if(action < 0):
				movement_cost -= 2
			for returns_loc1 in range(MAX_BOUND+1):
				for returns_loc2 in range(MAX_BOUND+1):
					prob_return1 = get_poission_prob(returns_loc1, mean_return_loc1)
					prob_return2 = get_poission_prob(returns_loc2, mean_return_loc2)
					cars_left1 = min(cars_loc1 - total_rentals_1 + returns_loc1 + action, max_cars)
					cars_left2 = min(cars_loc2 - total_rentals_2 + returns_loc2 - action, max_cars)
					if(cars_left1 < 0 or cars_left2 < 0):
						#not a valid state hence invalid action:
						continue
					#transition probability
					parking_cost = 0
					if(cars_left1 > 10):
						parking_cost += 4
					if(cars_left2 > 10):
						parking_cost += 4
					cars_left1 = int(cars_left1)
					cars_left2 = int(cars_left2)
					prob = prob_rental_1 * prob_rental_2 * prob_return1 * prob_return2
					step_reward = net_profit - movement_cost - parking_cost
					#print(returns)
					returns += (prob) * (step_reward + gamma * V[cars_left1, cars_left2])
	return returns
def policy_evaluation(V, policy):
	tol = 0.1
	while(True):
		delta = 0
		for i in range(max_cars+1):
			for j in range(max_cars+1):
				action = policy[i,j]
				oldV = V[i,j]
				newV = expected_return(V, action, (i,j))
				delta = max(delta, abs(oldV - newV))
				V[i,j] = newV
		if(delta < tol):
			break
		print("delta", delta)
	return V		

def policy_iteration():
	iter = 0
	while(True):
		iter += 1
		print(iter)
		policy_stable = True
		newV = policy_evaluation(val_state, policy)
		print(policy)
		displayp(policy, iter)
		displayv(newV, iter)

		for i in range(max_cars+1):
			for j in range(max_cars+1):
				old_action = policy[i,j]
				action_values = []
				for action in range(-5,6):
					q_s_a = expected_return(newV, action, (i,j))
					action_values.append(q_s_a)
				best_action = np.argmax(action_values) - 5
				if(old_action != best_action):
					policy_stable = False
				policy[i,j] = best_action
		
		if(policy_stable == True):
			break
policy_iteration()



