"""
max scheder
HW3 CSCI 5512
Value Iteration

Sequential Decision Problem
     ___ ___ ___ ___
   3|___|___|___|_+1|
   2|___|_X_|___|_-1|
   1|_s_|___|___|___|
      1   2   3   4

    Transition: 
        0.8 in desired direction (dd), 
        0.1 left of dd,
        0.1 right of dd.
    
    Discount Factor (γ):
        Preference for current rewards over future rewards
        γ = 0.99 (recommended by professor)
    
    Max_iterations:
        max_iters = 10,000 (recommended by professor)

    Non-terminal rewards:
        (i)     r = -2 [expectation: agent heads to closest exit, even -1]
        (ii)    r = -0.2 [expectation: ?]
        (iii)   r = -0.01 [expectation: no risks, heads away from -1 to not accidentally end there]
    
    P(s'|s, a) = probability of state s' (t+1) for action a in state s (t)
    Assumption: Markovian P(s'|s, a) depends only on s and not s-1, etc.
    Assumption: No fixed time limit, aka stationary policy
    Solution: Must specify what agent should do for any state that the agent might reach (aka a policy)
    Optimal: pi_star (pi*) policy that minimizes penalty / maximizes utility
"""
from random import random
from grid_world import Gw
import numpy as np
import sys
# init world

world_height = 3
world_width = 4
blocked_states = [(2,2)]
pos_exit_states = [(4,3)]
neg_exit_states = [(4,2)]
pos_exit_val = 1
neg_exit_val = -1
if sys.argv[1] != None:
    pain_of_existence = float(sys.argv[1])
else:   
    pain_of_existence = -0.01 
accessible_state_list = []
df = 0.99
randomness = 0.8

print(pain_of_existence)
world = Gw(world_height, world_width, blocked_states, pos_exit_states, neg_exit_states, pos_exit_val, neg_exit_val, pain_of_existence, accessible_state_list, df, randomness)
world.state_map()

def vi_alg(world, max_iters):
    value_star = {}
    policy_star = {}
    # print(world.pos_exit_states)
    # print(world.neg_exit_states)
    # set exits
    for state in world.pos_exit_states:
        # print(state)
        policy_star[state] = 'E'
    for state in world.neg_exit_states:
        policy_star[state] = 'E'
    # print(policy_star)
    
    # we are iterating max_iters time to find best policy rather than looking at a delta. 
    for iter in range(0, max_iters):
        # hit every state each iter
        for state in world.accessible_state_list:
            # print(state)
            # print(world.accessible_state_list)
            if state in world.pos_exit_states:
                potential_actions = ['E']
            elif state in world.neg_exit_states:
                potential_actions = ['E']
            else:
                potential_actions = ['W', 'A', 'S', 'D']

            value_star[state], policy_star[state] = max(action_options(world, value_star, state, potential_actions))
            # print(value_star, policy_star)
    
    p_star = []
    v_star = []
    for i in sorted (policy_star.items()) :
        p_star.append(i)

    for i in sorted (value_star.items()) :
        v_star.append(i)
    return v_star, p_star


def action_options(world, value_star, state, potential_actions):
    list_of_options = []
    for action in potential_actions:
        if action == 'E':
            if state in world.pos_exit_states:
                value_of_action = world.pos_exit_val
                # print(value_of_action)
            else:
                value_of_action = world.neg_exit_val
                # print(value_of_action)
        else:
            # print("here : not 'E'")
            intended_state = world.action(state, action)
            alt_states = world.unintended_action(state, action)
            
            # could also initialize dictionary with key for each state and value = 0 
            if intended_state not in value_star:
                value_star[intended_state] = 0
            if alt_states[0] not in value_star:
                value_star[alt_states[0]] = 0
            if alt_states[1] not in value_star:
                value_star[alt_states[1]] = 0

            intended_state_value = world.randomness * (world.pain_of_existence + value_star[intended_state] * world.df)
            alt_state_value0 = 0.5 * (1 - world.randomness) * (world.pain_of_existence + value_star[alt_states[0]] * world.df)
            alt_state_value1 = 0.5 * (1 - world.randomness) * (world.pain_of_existence + value_star[alt_states[1]] * world.df)
            
            value_of_action = intended_state_value + alt_state_value0 + alt_state_value1
        
        list_of_options.append((value_of_action, action))
    return list_of_options

values, policy = vi_alg(world, 10000)
print(values)
print("-------")
print(policy)