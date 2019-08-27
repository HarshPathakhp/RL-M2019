# Ex2.5
![Alt text](./ex2_5.png?raw=true "Step size vs sample avg. on non stationary environment")

Since the distributions of true rewards are changing with time, sample average faces a hard time adapting, because it's coefficent in the update rule for `Rn - Qn` is inversely proportional to number of times the bandit has been selected.
In contrast to this, using a constant step size weighs the future rewards with a constant factor everytime, and can adapt to a non stationary environment much faster than sample mean.

# Ex2.6 and Fig 2.3
![Alt text](./fig2_3_stationary.png?raw=true "Optimistic vs Realistic on stationary environment")

Initially, since we have selected highly optimistic estimates of reward for all the bandits, all bandits are going to get explored for some initial time, until the highly optimistic value has been diluted down, due to bandits that give less reward in practice. Since we have a bandit that promises the highest expected reward, there is going to be a high upward spike when this bandit gets selected. However, since even this bandit's reward is less compared to optimistic value, other bandits are going to be tried too. Hence, there is a spike in the initial phase.

![Alt text](./fig2_3_non-stationary.png?raw=true "Optimistic vs Realistic on stationary environment")
Optimistic initial values is not suitable for non-stationary environment, because its drive for exploration is inherently temporary. It will try out all the bandits in the beginning of time, but won't explore afterwards, since epsilon is 0 here.

# UCB Analysis
![Alt text](./ucb_stationary.png?raw=true "UCB")
In the stationary case, ucb gives higher expected reward than the rest. UCB performs better than e-greedy because e-greedy always picks the greedy action with probability `1-e+e/k`. UCB on the other hand, takes into account the potential of each bandit to be optimal by including the variance term in argmax calculation.
On the other hand, optimistic initial values fares better than e-greedy, since each bandit is going to get tried for some time initially and we will get better estimates than e-greedy. However, since this method only explores towards the beginning of time, it fares slightly worse than ucb.
![Alt text](./ucb_non-stationary.png?raw=true "UCB")
We know that ucb and optimistic initial values are not suitable for non-stationary environment. This is because both methods have been designed considering that the environment is stationary. Particularly, in UCB, the variance term will go down with increasing time steps, given that all bandits have been selected for large amount of times. Hence it's need for exploration will go down too and the method will become near-greedy. We already know that for optimistic initial values, its drive for exploration persists only in the beginning.

# Ex2.7
[a relative link](./rl_a1_q3.pdf)
