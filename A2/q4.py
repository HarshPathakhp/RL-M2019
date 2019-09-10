import numpy as np
""" 
grid representation (index)
 1  2  3  4  5
 6  7  8  9 10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
"""
def out_of_grid(ij):
    #checks whether an index (i,j) is outside the grid
    if(ij[0] < 0 or ij[0] > 4):
        return True
    if(ij[1] < 0 or ij[1] > 4):
        return True
    return False

def output_V(A):
    #print A
    for i in range(5):
        for j in range(5):
            print("%0.2f"%(A[i, j]), end = " ")
        print()

map_index_to_ij = {}
map_ij_to_index = {}
adjacency = {}
temp_count = 1
a = (0,1)
b = (0,3)
b_p = (2,3)
a_p = (4,1)

#create adjacency list
for i in range(5):
    for j in range(5):  
        adjacency[(i, j)] = [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]
        if(i == a[0] and j == a[1]):
            adjacency[(i, j)] = [a_p]
        elif(i == b[0] and j == b[1]):
            adjacency[(i, j)] = [b_p]
        map_index_to_ij[temp_count] = (i, j)
        map_ij_to_index[(i, j)] = temp_count
        temp_count += 1

"""
solving bellman optimality equation using linear programming
v*(s) = max (p(s',r|s,a)[r + yv*(s')])
since each s has 4 actions, we have 4 inequalities for each state
since there are 25 states, we have 100 inequalities
We now want to minimize sigma(v*(s)) with respect to these inequalities

Let c = e, where e is a vector of all 1's with length 25
A -> 100 x 25 matrix that will capture the inequalities
b -> 100 length vector that stores the constants -> r * p(s',r|s,a)
Optimization -> minimize cx such Ax <= b
where x is what we want to find (25 length vector storing all v(s))
"""
gamma = 0.9
A = np.zeros((100, 25))
B = np.zeros(100)
c = np.ones((25))
counter = 0
for idx in range(1, 26):
    ij = map_index_to_ij[idx]
    neigh = adjacency[ij]
    for i in range(len(neigh)):
        if(out_of_grid(neigh[i])):
            neigh[i] = ij
    for i in neigh:
        idx2 = map_ij_to_index[i]
        reward = None
        if(i == ij):
            reward = -1
        elif(ij == a):
            reward = 10
        elif(ij == b):
            reward = 5
        else:
            reward = 0
        coef = gamma
        const = -1*reward
        A[counter, idx - 1] += -1
        A[counter, idx2 - 1] += coef
        B[counter] += const
        counter += 1
    

from scipy.optimize import  linprog
res = linprog(c, A_ub=A, b_ub=B)
print(res)