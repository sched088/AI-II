"""
max scheder
HW3 CSCI 5512
Value Iteration from scratch

Sequential Decision Problem
     ___ ___ ___ ___
   3|___|___|___|_+1|
   2|___|_X_|___|_-1|
   1|_s_|___|___|___|
      1   2   3   4

    Non-terminal rewards:
        (i)     r = -2 [expectation: agent heads to closest exit, even -1]
        (ii)    r = -0.2 [expectation: ?]
        (iii)   r = -0.01 [expectation: no risks, heads away from -1 to not accidentally end there]

    Transition: 
        0.8 in desired direction (dd), 
        0.1 left of dd,
        0.1 right of dd.
    
    Discount Factor (γ):
        Preference for current rewards over future rewards
        γ = 0.99 (recommended by professor)
"""

from random import random


class Gw:

    # Class defined variables (as given in HW3 prompt). Allows to be overwritten if desired
    # world_height = 3
    # world_width = 4
    # blocked_states = [(2,2)]
    # pos_exit_states = [(3,4)]
    # neg_exit_states = [(2,4)]
    # pos_exit_val = 1
    # neg_exit_val = -1
    # pain_of_existence = -0.01
    # accessible_state_list = []

    def __init__(self, world_height, world_width, blocked_states, pos_exit_states, neg_exit_states, pos_exit_val, neg_exit_val, pain_of_existence, accessible_state_list, df, randomness):
        # integer > 0 (represents rows in world)
        self.world_height = world_height
        # integer > 0 (represents columns in world)
        self.world_width = world_width
        # list of tuples, each tuple represents a coordinate that the agent cannot access
        self.blocked_states = blocked_states
        # list of tuples, each tuple represents a coordinate that the agent can exit the world with a positive reward
        self.pos_exit_states = pos_exit_states
        # list of tuples, each tuple represents a coordinate that the agent can exit the world with a negative reward
        self.neg_exit_states = neg_exit_states
        # integer > 0 (reward value of exiting world at positive locations)
        self.pos_exit_val = pos_exit_val
        # integer < 0 (reward value of exiting world at negative locations)
        self.neg_exit_val = neg_exit_val
        # cost of agent being in a state
        self.pain_of_existence = pain_of_existence
        # Initialize empty state list
        self.accessible_state_list = accessible_state_list
        # Initialize discount factor
        self.df = df
        # Initialize 1 - probability of agent moving sum(left or right) 
        self.randomness = randomness

    # Create the world from init variables
    def state_map(self):
        state_list = []
        for x_grid in range(1, self.world_width + 1):
            for y_grid in range(1, self.world_height + 1):
                state_list.append((x_grid, y_grid))
        
        # remove blocked states
        for blocked_state in self.blocked_states:
            state_list.pop(state_list.index(blocked_state))
            self.accessible_state_list = state_list

        # print("here")
        # for state in self.accessible_state_list:
        #     print("accessible_states: " + str(state))
        return self.accessible_state_list

    def print(self):
        for state in self.accessible_state_list:
            print("accessible_states: " + str(state))
        print("here")

    def unintended_action(self, state, action):
        if action == 'W':
            return [self.action(state, 'A'), self.action(state, 'D')]
        if action == 'A':
            return [self.action(state, 'S'), self.action(state, 'W')]
        if action == 'S':
            return [self.action(state, 'D'), self.action(state, 'A')]
        if action == 'D':
            return [self.action(state, 'W'), self.action(state, 'S')]    
            
    def intent_to_action(self, intended_action):
        if intended_action == 'E':
            self.action('E')
        decider = random.random()
        if decider < 0.8:
            self.action(intended_action)
        if 0.8 >= decider < 0.9: # left
            if intended_action == 'W':
                self.action('A')
            if intended_action == 'S':
                self.action('D')
            if intended_action == 'A':
                self.action('S')
            if intended_action == 'D':
                self.action('W')
        if decider >= 0.9: # right
            if intended_action == 'W':
                self.action('D')
            if intended_action == 'S':
                self.action('A')
            if intended_action == 'A':
                self.action('W')
            if intended_action == 'D':
                self.action('S')
        
    # Define how agent can move
    # Note game format: W=up, A=left , S=down, D=right 
    def action(self, state, move):
        if move == 'E':
            return 'E'
        if move == 'W':
            new_state = (state[0], state[1]+1)
        if move == 'S':
            new_state = (state[0], state[1]-1)
        if move == 'A':
            new_state = (state[0]-1, state[1])
        if move == 'D':
            new_state = (state[0]+1, state[1]) 
        
        if new_state in self.accessible_state_list:
            # update state if possible
            return new_state
        else:
            # output same state if unable to move
            return state

# p = grid_world()
# print(p.state_map())

class Agent(Gw):
    # Class defined variables. Allows to be overwritten if desired
    df = 0.99
    prob_desired_action = 0.8
    prob_left_action = 0.1
    prob_right_action = 0.1
    # Set starting location
    state = (1, 1)
    action_list = ['W', 'A', 'S', 'D']

    # Set discount factor (df) 
    def __init__(self, df, prob_desired_action, prob_left_action, prob_right_action, state, action_list):
        self.df = df
        self.prob_desired_action = prob_desired_action
        self.prob_left_action = prob_left_action
        self.prob_right_action = prob_right_action
        self.state = state
        self.action_list = action_list

    # In terminal state?
    # def exit_state_check(self):
    #     if self.state == self.pos_exit_states:
    #         self.exit_state = True
    #     if self.state == self.neg_exit_states:
    #         self.exit_state = True
    #     return self.exit_state

    # Address transition probability
    def intent_to_action(self, intended_action):
        if intended_action == 'E':
            self.action('E')
        decider = random.random()
        if decider < 0.8:
            self.action(intended_action)
        if 0.8 >= decider < 0.9: # left
            if intended_action == 'W':
                self.action('A')
            if intended_action == 'S':
                self.action('D')
            if intended_action == 'A':
                self.action('S')
            if intended_action == 'D':
                self.action('W')
        if decider >= 0.9: # right
            if intended_action == 'W':
                self.action('D')
            if intended_action == 'S':
                self.action('A')
            if intended_action == 'A':
                self.action('W')
            if intended_action == 'D':
                self.action('S')

    # Define how agent can move
    # Note game format: W=up, A=left , S=down, D=right 
    def action(self, move):
        if move == 'E':
            return 'E'
        if move == 'W':
            new_state = (self.state[0], self.state[1]+1)
        if move == 'S':
            new_state = (self.state[0], self.state[1]-1)
        if move == 'A':
            new_state = (self.state[0]-1, self.state[1])
        if move == 'D':
            new_state = (self.state[0]+1, self.state[1]) 
        
        if new_state in self.accessible_state_list:
            # update state if possible
            return new_state
        else:
            # output same state if unable to move
            return self.state