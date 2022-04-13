"""
5521 AI II HW1 Programming part
Max Scheder
Sprint 2022

Gibbs sampler MCMC
"""

"""
'rain network'
Evidence Variables
s = 'Sprinker = True' 
w = 'WetGrass = True'

Other Variables
r = 'Rain = True'
"""

"""
                    Cloudy
                     |   |
        Sprinker <---+   +--->  Rain
        |                        |
        +------>  WetGrass <-----+

CPTs
    Cloudy          Sprinker        Rain            WetGrass     
    P(C) = 0.5      C | P(S|C)      C | P(R|C)      S R | P(W|S,R)
                    T | 0.10        T | 0.80        T T | 0.99
                    F | 0.50        F | 0.20        T F | 0.90
                                                    F T | 0.90
                                                    F F | 0.01
"""

import random
# Part A: Store the conditional probabilities shown below.
# For Gibbs Need: P(c | r, w, s), P(c | ¬r, w, s), P(r | c, w, s), P(r | ¬c, w, s), 
#                 P(¬c | r, w, s), P(¬c | ¬r, w, s), P(¬r | c, w, s), P(¬r | ¬c, w, s), 
# Markov Blanket = MB = node's parents, children, and the parents of its children (self excluded)

# # P(c | r, w, s) = P(r, w, s | c) * P(C) / P(r, w, s)
# # P(c, r, w, s) = P(C) * P(R|C) * P(w|s, R) * P(s|C)
# # P(r, w, s) = P(R|C) * P(w|s, R) * P(s|C)
# # P(c | r, w, s) = P(C) = 0.5

# P(c | r, w, s) => P(c | r, s)
pcr = 0.5 * 0.8 * 0.1
# P(c | ¬r, w, s) => P(c | ¬r, s )
pcnr = 0.5 * 0.2 * 0.1

# P(¬c | r, w, s) => # P(¬c | r, s)
pncr = 0.5 * 0.2 * 0.5
# P(¬c | ¬r, w, s) => P(¬c | ¬r, s) 
pncnr = 0.5 * 0.8 * 0.5

# P(r | c, w, s)
prc = pcr
# P(r | ¬c, w, s) 
prnc = 0.2 * 0.5 * 0.99 * 0.5

# P(¬r | c, w, s)
pnrc = 0.2 * 0.5 * 0.9 * 0.1
# P(¬r | ¬c, w, s)
pnrnc = 0.8 * 0.5 * 0.9 * 0.5

# print(str(pcr + pcnr + prc + prnc + pncr + pncnr + pnrc + pnrnc))
# Part B: Estimate P(r |s, w) using Gibbs Sampling for 100 and 10,000 steps. 
from re import T
from sys import stdin


# num_steps = 100
# num_steps = 10000
num_steps = int(input())

def Gibbs(num_steps):
    state_list = []
    S = 1
    W = 1
    # Initialize Clour and Rain randomly
    C = random.randint(0,1)
    R = random.randint(0,1)
    # Initial State
    state = [C, S, R, W]

    # Sample non-evidence variables repeatedly and in arbitrary order
    for step in range(num_steps - 1):
        # print("state" + str(state))
        state_list.append(state[:])
        i = random.random()
        if random.randint(0,1) == 1: # Sample C
            if state[2] == 0:
                # print("1")
                # print(i, (pncr/(pncr + pncnr)))
                if i <= (pncr/(pncr + pncnr)):
                    # print("A")
                    state[0] = 1
                else:
                    # print("B")
                    state[0] = 0
            else:
                # print("2")
                # print(i, (pcr/(pcr + pncr)))
                if i <= (pcr/(pcr + pncr)):
                    state[0] = 1
                else:
                    state[0] = 0
        else: # Sample R
            if state[0] == 0:
                # print("3")
                # print(i, (prnc/(prnc + pnrnc)))
                if i <= (prnc/(prnc + pnrnc)):
                    # print("A)")
                    state[2] = 1
                else:
                    # print("B")
                    state[2] = 0
            else:
                # print("4")
                # print(i, (prc/(prc + pnrc)))
                if i <= (prc/(prc + pnrc)):
                    # print("A")
                    state[2] = 1
                else:
                    # print("B")
                    state[2] = 0
        # print(state)
    # Count up to get P(r | s, w)
        # Note: This can be combined into the for step in num_steps loop. 
    count = 0 
    # print(state_list)
    for state in state_list:
        if state[2] == 1:
            count += 1
    return(count/num_steps)

print(Gibbs(num_steps))