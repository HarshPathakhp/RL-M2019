from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
"""
About game and implementation -

"""
cards = [1,2,3,4,5,6,7,8,9,10,10,10,10] #1 ace, 2-10 cards and 3 face cards
ACTION_HIT = 0
ACTION_STICK = 1

#state -> sum(12-21), usable_ace -> boolean, dealer's show card -> (A-10)
#player policy that sticks for sum 20 and 21 and hits otherwise
player_policy = np.zeros((22))
player_policy[20] = ACTION_STICK
player_policy[21] = ACTION_STICK

class Blackjack:
	def __init__(self):
		self.dealer_policy = np.zeros((22)) 
		for i in range(17,22):
			self.dealer_policy[i] = ACTION_STICK
	@staticmethod
	def get_card():
		return np.random.choice(cards)
	@staticmethod
	def get_card_value(card:int):
		return 11 if card == 1 else card
	def run_episode(self, policy_player, initial_state = None, initial_action = None):
		#variables
		player_sum = 0
		usable_ace_player = False
		dealer_card1 = 0
		dealer_card2 = 0
		usable_ace_dealer = False
		sa_sequence = []
		if initial_state is None:
			while player_sum < 12:
				card = Blackjack.get_card()
				player_sum += Blackjack.get_card_value(card)
				if(player_sum > 21):
					#this card must be an ace
					player_sum -= 10
				else:
					usable_ace_player |= (card == 1)
			dealer_card1 = Blackjack.get_card()
			dealer_card2 = Blackjack.get_card()
		else:
			player_sum, usable_ace_player, dealer_card1 = initial_state
			dealer_card2 = Blackjack.get_card()
		usable_ace_dealer = 1 in (dealer_card1, dealer_card2)
		dealer_sum = Blackjack.get_card_value(dealer_card1) + Blackjack.get_card_value(dealer_card2)
		if(dealer_sum == 22):
			dealer_sum -= 10
		assert dealer_sum <= 21
		assert player_sum <= 21

		#player's turn
		while True:
			if(initial_action is not None):
				action = initial_action
				initial_action = None
			else:
				action = policy_player(player_sum, usable_ace_player, dealer_card1)		
			sa_sequence.append([(player_sum, usable_ace_player, dealer_card1), action])
			if action == ACTION_STICK:
				break
			#action is hit

			newcard = Blackjack.get_card()
			newvalue = Blackjack.get_card_value(newcard)
			ace_count = int(usable_ace_player)
			if newcard == 1:
				ace_count += 1
			player_sum += newvalue
			while player_sum > 21 and ace_count:
				player_sum -= 10
				ace_count -= 1
			if player_sum > 21:
				return sa_sequence, -1
			usable_ace_player = (ace_count == 1)

		#dealer's turn
		while True:
			action = self.dealer_policy[dealer_sum]
			if action == ACTION_STICK:
				break
			newcard = Blackjack.get_card()
			newvalue = Blackjack.get_card_value(newcard)
			ace_count = int(usable_ace_dealer)
			if newcard == 1:
				ace_count += 1
			dealer_sum += newvalue
			while dealer_sum > 21 and ace_count:
				dealer_sum -= 10
				ace_count -= 1
			if dealer_sum > 21:
				return sa_sequence, 1
			usable_ace_dealer = (ace_count == 1)
		assert player_sum <= 21 and dealer_sum <= 21
		if player_sum < dealer_sum:
			return sa_sequence, -1
		elif player_sum == dealer_sum:
			return sa_sequence, 0
		return sa_sequence, 1

def policy_prediction(episodes, target_policy):
	values_usable_ace = np.zeros((22,11)) #index i -> sum on player's cards, index j -> dealer's show card
	values_non_usable_ace = np.zeros((22,11))
	counts_usable_ace = np.zeros((22,11))
	counts_non_usable_ace = np.zeros((22,11))
	blackjack = Blackjack()
	for eps in tqdm(range(episodes)):
		seq, reward = blackjack.run_episode(target_policy, initial_state = None, initial_action = None)
		for (psum, ace, dealercard), _ in seq:
			if(ace):
				counts_usable_ace[psum, dealercard] += 1
				values_usable_ace[psum, dealercard] += reward
			else:
				counts_non_usable_ace[psum, dealercard] += 1
				values_non_usable_ace[psum, dealercard] += reward
	counts_usable_ace[counts_usable_ace == 0] = 1
	counts_non_usable_ace[counts_non_usable_ace == 0] = 1
	return values_usable_ace/counts_usable_ace, values_non_usable_ace/counts_non_usable_ace

def mc_es(episodes, target_policy):
	state_action_values = np.zeros((22,11,2,2))
	state_action_count = np.ones((22,11,2,2))
	def behavior_policy(player_sum, usable_ace, dealercard):
		avg_state_action_values = state_action_values[player_sum, dealercard, int(usable_ace), :] / state_action_count[player_sum, dealercard, int(usable_ace), :]
		return np.argmax(avg_state_action_values)
	blackjack = Blackjack()
	for eps in tqdm(range(episodes)):
		policy_to_follow = behavior_policy if eps else target_policy	
		initial_action = np.random.choice([0,1])
		initial_state = [np.random.choice(range(12,22)),
						bool(np.random.choice([0, 1])),
						np.random.choice(range(1, 11))]
		seq, reward = blackjack.run_episode(policy_to_follow, initial_state = initial_state, initial_action = initial_action)
		for (psum, ace, dealercard), action in seq:
			state_action_values[psum, dealercard, int(ace), int(action)] += reward
			state_action_count[psum, dealercard, int(ace), int(action)] += 1
	return state_action_values/state_action_count
def off_policy(target_policy, behavior_policy, eps = 10000):
	initial_state = [13, True, 2]
	ordinary = None
	weighted = None
	bj = Blackjack()
	sampling_ratios = []
	returns = []
	for ep in range(eps):
		num = 1
		den = 1
		seq, reward = bj.run_episode(behavior_policy, initial_state = initial_state)
		for (psum, ace, dealercard), action in seq:
			if(action == target_policy(psum, ace, dealercard)):
				den *= 0.5
			else:
				num = 0
				break
		sampling_ratios.append(num/den)
		returns.append(reward) #gamma is 1
	sampling_ratios = np.array(sampling_ratios)
	returns = np.array(returns)
	weighted_sum = sampling_ratios * returns
	weighted_returns = np.add.accumulate(weighted_sum)
	sampling_ratios = np.add.accumulate(sampling_ratios)
	ordinary_sampling = weighted_returns /  np.arange(1, eps+1)
	with np.errstate(divide='ignore',invalid='ignore'):
		weighted_sampling = np.where(sampling_ratios != 0, weighted_returns / sampling_ratios, 0)
	return ordinary_sampling, weighted_sampling
def fig5_1():
	
	def target_policy(player_sum, usable_ace, showcard):
		return player_policy[player_sum]

	values_usable_ace_10k, values_non_usable_ace10k = policy_prediction(int(1e4), target_policy)
	values_usable_ace_500k, values_non_usable_ace500k = policy_prediction(int(5e5), target_policy)

	states = [values_usable_ace_10k[12:,1:],
			values_usable_ace_500k[12:,1:],
			values_non_usable_ace10k[12:,1:],
			values_non_usable_ace500k[12:,1:]]

	titles = ['Usable Ace, 10000 Episodes',
    'Usable Ace, 500000 Episodes',
    'No Usable Ace, 10000 Episodes',
    'No Usable Ace, 500000 Episodes']

	_, axes = plt.subplots(2, 2, figsize=(40, 30))
	plt.subplots_adjust(wspace=0.1, hspace=0.2)
	axes = axes.flatten()

	for state, title, axis in zip(states, titles, axes):
		fig = sns.heatmap(np.flipud(state), cmap="jet", ax=axis, xticklabels=range(1, 11),
			yticklabels=list(reversed(range(12, 22))))
		fig.set_ylabel('player sum', fontsize=30)
		fig.set_xlabel('dealer showing', fontsize=30)
		fig.set_title(title, fontsize=30)

	plt.savefig('./figure_5_1.png')
	plt.close()

def fig5_2():
	def target_policy(player_sum, usable_ace, showcard):
		return player_policy[player_sum]

	qsa500k = mc_es(int(5e5), target_policy)
	values_usable_ace = np.max(qsa500k[12:,1:,1,:], axis = -1)
	values_non_usable_ace = np.max(qsa500k[12:,1:,0,:], axis = -1)

	policy_optimal_usable_ace = np.argmax(qsa500k[12:,1:,1,:], axis = -1)
	policy_optimal_non_usable_ace = np.argmax(qsa500k[12:,1:,0,:], axis = -1)

	images = [policy_optimal_usable_ace,
		values_usable_ace,
		policy_optimal_non_usable_ace,
		values_non_usable_ace]

	titles = ['Optimal policy with usable Ace',
		'Optimal value with usable Ace',
		'Optimal policy without usable Ace',
		'Optimal value without usable Ace']

	_, axes = plt.subplots(2, 2, figsize=(40, 30))
	plt.subplots_adjust(wspace=0.1, hspace=0.2)
	axes = axes.flatten()

	for image, title, axis in zip(images, titles, axes):
		fig = sns.heatmap(np.flipud(image), cmap="jet", ax=axis, xticklabels=range(1, 11),
							yticklabels=list(reversed(range(12, 22))))
		fig.set_ylabel('player sum', fontsize=30)
		fig.set_xlabel('dealer showing', fontsize=30)
		fig.set_title(title, fontsize=30)

	plt.savefig('./figure_5_2.png')
	plt.close()
def fig5_3():
	def target_policy(player_sum, usable_ace, showcard):
		return player_policy[player_sum]
	def behavior_policy(player_sum, usuable_ace, showcard):
		x = np.random.binomial(1,0.5) #output is 0(hit) or 1(stick) with equal probability
		return x
	true_value = -0.27726
	episodes = 10000
	eo = np.zeros(episodes)
	ew = np.zeros(episodes)
	runs = 100
	for run in tqdm(range(runs)):
		o, w = off_policy(target_policy, behavior_policy)
		eo += np.power(o - true_value, 2)
		ew += np.power(w - true_value, 2) 
	eo /= runs
	ew /= runs	
	plt.plot(eo, label='Ordinary Importance Sampling')
	plt.plot(ew, label='Weighted Importance Sampling')
	plt.xlabel('Episodes (log scale)')
	plt.ylabel('Mean square error')
	plt.xscale('log')
	plt.legend()
	plt.savefig('./fig_5_3.png')
	plt.close()

if __name__ == '__main__':
	#fig5_1()
	#fig5_2()
	fig5_3()