import numpy as np
import matplotlib.pyplot as plt
class Testbed:
    def __init__(self, real_reward = 0, time_mean = 0, time_variance = 0.01, steps = int(1e4), init_estimates = 0,
                eps = 0.1, alpha = 0.1, use_alpha = True, use_ucb = False, ucb_param = 2):
        
        self.num_bandits = 10
        self.time = 0
        self.use_ucb = use_ucb
        self.ucb_param = ucb_param

        #keep action counts of each arm
        self.action_counts = np.zeros((self.num_bandits))
        self.real_reward = real_reward

        #self.true_action_val is non stationary
        self.true_action_val = np.array([real_reward for i in range(self.num_bandits)], dtype = np.float64)
        #self.true_action_val += np.random.normal(loc = 0, scale = 1, size = self.num_bandits)
        
        #non-stationary setting
        self.time_mean = time_mean
        self.time_variance = time_variance
        
        #estimated expected reward function
        self.q_estimates = np.array([init_estimates for i in range(self.num_bandits)], dtype = np.float64)
        self.init_estimates = init_estimates
        
        #process variables
        self.steps = steps
        self.eps = eps #e-greedy epsilon
        self.alpha = alpha #step size 
        self.use_alpha = use_alpha #if false, uses sample mean instead
        
        self.indices = np.arange(self.num_bandits)
    
    def select(self):
        #selects the arm using e-greedy method
        toss = np.random.uniform()
        rewards_list = self.true_action_val + np.random.normal(loc = 0, scale = 1, size = self.num_bandits)
        if(toss < self.eps):
            ch = np.random.choice(self.indices)
            return ch, rewards_list[ch]
        if(self.use_ucb):
            ucb_estimates = self.q_estimates + np.sqrt(np.log(self.time + 1) / (self.action_counts + 1e-5))
            max_idx = np.argmax(ucb_estimates)
            return max_idx, rewards_list[max_idx]
        else:
            max_idx = np.argmax(self.q_estimates)
            return max_idx, rewards_list[max_idx]
    
    def update(self, arm_id : int, step_reward):
        self.time += 1
        self.action_counts[arm_id] += 1
        if self.use_alpha:
            self.q_estimates[arm_id] += self.alpha * (step_reward - self.q_estimates[arm_id])
        else:
            self.q_estimates[arm_id] += (step_reward - self.q_estimates[arm_id])/self.action_counts[arm_id]
        
    def reset(self):
        self.time = 0
        self.action_counts = np.zeros((self.num_bandits))
        self.true_action_val = np.array([self.real_reward for i in range(self.num_bandits)], dtype = np.float64)
        #self.true_action_val += np.random.normal(loc = 0, scale = 1, size = self.num_bandits)
        self.q_estimates = np.array([self.init_estimates for i in range(self.num_bandits)], dtype = np.float64)
        
    def non_stationary(self):
        step = np.random.normal(loc = self.time_mean, scale=self.time_variance, size = self.num_bandits)
        self.true_action_val = step + self.true_action_val
        
    def start(self, mode, runs = 1000):
        c = 0
        rewards = np.zeros((runs, self.steps), dtype = np.float64)
        best_action_counts = np.zeros((runs, self.steps), dtype = np.float64)
        for rid in range(runs):
            print(rid)
            self.reset()
            for tstep in range(self.steps):
                arm_id, step_reward = self.select()
                rewards[rid, tstep] = step_reward
                self.update(arm_id, step_reward)
                if(arm_id == np.argmax(self.true_action_val)):
                    c += 1
                    best_action_counts[rid, tstep] = 1
                if(mode == 1):
                    self.non_stationary()
        
        print(self.action_counts)
        print(self.action_counts)
        print(self.true_action_val)
        print(self.q_estimates)
        print(c)
        return rewards.mean(axis = 0), best_action_counts.mean(axis = 0)
def ex_ucb(mode = 0):
    #0 -> stationart
    #1 -> non-stationary
    use_alpha = True
    if(mode == 0):
        use_alpha = False
    t1 = Testbed(use_alpha = use_alpha, init_estimates = 0, eps = 0, use_ucb = True)
    r1,_ = t1.start(mode)
    t2 = Testbed(use_alpha = use_alpha, init_estimates = 5, eps = 0)
    r2,_ = t2.start(mode)
    t3 = Testbed(use_alpha = use_alpha, init_estimates = 0, eps = 0.1)
    r3,_ = t3.start(mode)
    plt.figure(figsize = (10,10))
    plt.plot(r1, label = "ucb, c = 2, alpha = 0.1, eps = 0")
    plt.plot(r2, label = "optimistic = 5, eps = 0, alpha = 0.1")
    plt.plot(r3, label = "realistic, eps = 0.1")
    plt.xlabel("steps")
    plt.ylabel("avg reward")
    plt.legend()
    string = "stationary" if mode == 0 else "non-stationary"
    plt.savefig("ucb_%s.png" %(string))
    plt.close()


def fig2_3(mode = 0):
    # 0 -> stationary
    # 1 -> non-stationary
    t1 = Testbed(use_alpha = True, init_estimates = 5, eps = 0)
    rewards1, best_action1 = t1.start(mode)
    t2 = Testbed(use_alpha = True)
    rewards2, best_action2 = t2.start(mode)
    plt.figure(figsize = (10,10))
    plt.plot(best_action1, label = "optimistic = 5, eps = 0, alpha = 0.1")
    plt.plot(best_action2, label = "relaistic = 0, eps = 0.1, alpha = 0.1")
    plt.xlabel("steps")
    plt.ylabel("% optimal action")
    plt.legend()
    string = "stationary" if mode == 0 else "non-stationary"
    plt.savefig("fig2_3_%s.png" %(string))
    plt.close()

def ex2_5(mode = 0):
    t1 = Testbed(use_alpha = True)
    rewards1, best_action1 = t1.start(mode)
    t2 = Testbed(use_alpha = False)
    rewards2, best_action2 = t2.start(mode)
    plt.figure(figsize = (10,20))
    plt.subplot(2,1,1)
    plt.plot(rewards1, label = "alpha = 0.1")
    plt.plot(rewards2, label = "sample average")
    plt.xlabel("steps")
    plt.ylabel("avg. reward")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(best_action1, label = "alpha = 0.1")
    plt.plot(best_action2, label = "sample mean")
    plt.xlabel("steps")
    plt.ylabel("% optimal action")
    plt.legend()
    
    plt.savefig("ex2_5.png")
    plt.close()
if __name__ == "__main__":
    #ex_ucb(1)
    #fig2_3(mode =  0)
    ex2_5(mode = 1)
    #fig2_3(mode =  1)