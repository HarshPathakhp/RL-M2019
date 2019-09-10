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

def output_A(A):
    #print A
    for i in range(25):
        for j in range(25):
            print(A[i, j], end = " ")
        print()
def output_c(c):
    #print c
    for i in range(25):
        print(c[i])

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

""" solving Ax = c """
A = np.zeros((25, 25))
c = np.zeros((25))
gamma = 0.9

for idx in range(1, 26):
    #bellman equation for state idx under equiprobable policy
    ij_index = map_index_to_ij[idx]
    neighbors = adjacency[ij_index]
    A[idx-1,idx - 1] = 1
    c_idx = 0
    for nij in  neighbors:
        reward = None
        if(out_of_grid(nij)):
            reward = -1
        elif(ij_index == a):
            reward = 10
        elif(ij_index == b):
            reward = 5  
        else:
            reward = 0
        
        equi_chance = 1/(len(neighbors)) #p(a|s)
        const = equi_chance * reward #(r * p(a|s))
        dependency_idx = None #neighbour idx from 1 to 25
        if(out_of_grid(nij)):
            dependency_idx = idx
        else:
            dependency_idx = map_ij_to_index[nij]
        dependency_coef = -1 * (gamma * equi_chance) #gamma * p(a|s), multiply by -1 to send to left side
        A[idx - 1, dependency_idx - 1] += dependency_coef
        c_idx += const #add up const for all neighbors
    c[idx - 1] = c_idx  

""" 
Ax = c
x = A(-1)c
"""
Ainv = np.linalg.inv(A)
v_vector = np.matmul(Ainv, c)

#print expected returns
for i in range(25):
    if((i+1) % 5 == 0):
        print("%1.2f" %(v_vector[i]))
    else:
        print("%1.2f" %(v_vector[i]), end = " ")