import numpy as np
import os
"""
grid representation
 x  1  2  3 
 4  5  6  7
 8  9 10 11
12 13 14  x
"""
ij_idx = {}
idx_ij = {}
adj = {}
def utility():
    global ij_idx
    global idx_ij
    global adj
    idx = 1
    for i in range(4):
        for j in range(4):
            if((i == 0 and j == 0) or (i == 3 and j == 3)):
                continue
            adj[(i, j)] = [(i-1,j), (i,j-1), (i+1,j), (i,j+1)]
            ij_idx[(i,j)] = idx
            idx_ij[idx] = (i,j)
            idx += 1   
utility()

def out_of_grid(ij):
    #checks whether an index (i,j) is outside the grid
    if(ij[0] < 0 or ij[0] > 3):
        return True
    if(ij[1] < 0 or ij[1] > 3):
        return True
    return False

def value_iteration():
    dir = "./policy&value_q4"
    os.makedirs(dir, exist_ok=True)
    f = open(os.path.join(dir, "value_iter.log"), 'w')
    V = np.zeros((4,4))
    tolerance = 0.01
    iter = 0
    while(True):
        iter += 1
        delta = 0
        for id in range(1, 15):
            ij = idx_ij[id]
            neighbors = adj[ij]
            reward = -1
            max_candidate = -1e9
            oldV = V[ij]
            for nij in neighbors:
                point = nij
                if(out_of_grid(nij)):
                    point = ij
                max_candidate = max(max_candidate, reward + V[point])
            V[ij] = max_candidate
            delta = max(delta, abs(max_candidate - oldV))
        f.write("iter %d\n"%(iter))
        for i in range(4):
            for j in range(4):
                f.write("%0.2f "%(V[i,j]))
            f.write("\n")
        f.write("-"*20 + "\n")
        if(delta < tolerance):
            break    


    f.close()
def get_state(ij, action):
    """
    0 for up
    1 for left
    2 for down
    3 for right
    """
    
    if(action == 0):
        return (ij[0]-1,ij[1])
    elif(action == 1):
        return (ij[0],ij[1]-1)
    elif(action == 2):
        return (ij[0]+1,ij[1])
    elif(action == 3):
        return (ij[0],ij[1]+1)

def policy_evaluation(V, policy):
    tol = 0.01
    iters = 0
    while(True):
        iters += 1
        delta = 0
        for id in range(1, 15):
            ij = idx_ij[id]
            i = ij[0]
            j = ij[1]
            oldV = V[ij]
            estimate = 0
            for action in range(4):
                point = ij
                nij = get_state(ij, action)
                if(not out_of_grid(nij)):
                    point = nij
                estimate += (policy[id-1, action] * (-1 + V[point]))
            delta = max(delta, abs(oldV - estimate))
            V[ij] = estimate
        if(delta < tol):
            break
    #print(iters)              
    return V        
def logger(f, policy):
    f.write("X ")
    when = [2, 6, 10]
    for i in range(14):
        action = np.argmax(policy[i])
        if(action == 0):
            f.write("up ")
        elif(action == 1):
            f.write("left ")
        elif(action == 2):
            f.write("down ")
        else:
            f.write("right ")
        if(i in when):
            f.write("\n")    
    f.write("X\n")

def policy_iteration():
    dir = "./policy&value_q4"
    os.makedirs(dir, exist_ok=True)
    f = open(os.path.join(dir, "policye_iter.log"), 'w')
    
    V = np.zeros((4,4))
    policy = np.ones((14, 4))/4
    iter = 0
    f.write("iter %d\n"%(iter))
    for i in range(4):
        for j in range(4):
            f.write("%0.2f "%(V[i,j]))
        f.write("\n")
    logger(f, policy)
    f.write("-"*20 + "\n")
        
    while(True):
        iter += 1
        V = policy_evaluation(V, policy)
        f.write("iter %d\n"%(iter))
        for i in range(4):
            for j in range(4):
                f.write("%0.2f "%(V[i,j]))
            f.write("\n")
        policy_stable = True
        for id in range(1, 15):
            ij = idx_ij[id]
            old_argmax = np.argmax(policy[id-1])
            #fixing bug in 4.4
            #----
            old_best_V = V[ij]
            #----
            q_s_a = np.zeros((4))
            for action in range(4):
                new_state = get_state(ij, action)
                if(out_of_grid(new_state)):
                    new_state = ij
                q_s_a[action] = -1 + V[new_state]
            new_argmax = np.argmax(q_s_a)
            #fixing bug in 4.4
            new_best = np.max(q_s_a)
            #----
            if(old_argmax != new_argmax):
                #fixing bug in 4.4
                if(new_best > old_best_V):
                    policy_stable = False
                #-----
            policy[id-1] = np.eye(4)[new_argmax]
        logger(f, policy)
        f.write("-"*20 + "\n")
        
        if policy_stable:
            break
    
    #print(V)
value_iteration()
policy_iteration()