import argparse
import random
from random import choices
import sys
import os
import copy

# ----------GLOBAL VARS----------
import numpy as np

states = []
actions = []
result_states = []
result_probabilities = []

# Building parser to get input values
parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args() 
with open(args.filename) as f:
# with open(os.path.join(sys.path[0], "assignment2test"), "r") as f:
    # split input into the 3 sections
    sections = f.read().split("\n")
    count = 0
    for line in sections:
        parts = line.split('/')
        if parts[0] not in states:
            states.append(parts[0])
            actions.append([])
            result_states.append([])
            result_probabilities.append([])
    for line in sections:
        parts = line.split('/')
        Index = states.index(parts[0])
        if parts[1] not in actions[Index]:
            actions[Index].append(parts[1])
            result_states[Index].append([])
            result_probabilities[Index].append([])
            #print(Index,actions[Index].index(parts[1]))
        if parts[2] not in result_states[Index][actions[Index].index(parts[1])]:
            result_states[Index][actions[Index].index(parts[1])].append(parts[2])
            result_probabilities[Index][actions[Index].index(parts[1])].append(float(parts[3]))

states.append("In")
actions.append(["Putt"])
result_states.append([["In"]])
result_probabilities.append([[1]])

q_table = copy.deepcopy(actions)
for i in range(len(q_table)):
    for j in range(len(q_table[i])):
        q_table[i][j] = 0

optimal_policy = []
learning_rate = 0.1
discount = 0.9
episodes = 5000

i=0

while i < episodes:
    done = False
    starting_state = "Fairway"
    current_state = "Fairway"
    epsilon = 0.5 # to determine exploration/exploitation
    start_epsilon_decaying = 1
    end_epsilon_decaying = episodes / 2
    epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)
    while not done:
        index_of_state = states.index(current_state)
        if np.random.random() > epsilon: # exploitation
            min = np.min(q_table[index_of_state])
            index_of_minimum_q =q_table[index_of_state].index(min)
            action = actions[index_of_state][index_of_minimum_q]
        else: # exploration
            action = random.choice(actions[index_of_state])
        index_of_action = actions[index_of_state].index(action)
        result_state = random.choices(result_states[index_of_state][index_of_action],
                                     result_probabilities[index_of_state][index_of_action]) # getting the resulting state from the action's possible resulting states and probabilities
        index_of_result_state = states.index(result_state[0])
        future_q = np.min(q_table[index_of_result_state])
        current_q = q_table[index_of_state][index_of_action]
        new_q = (1 - learning_rate) * current_q + learning_rate*(1+discount * future_q) # calculate new q values
        q_table[index_of_state][index_of_action] = new_q
        current_state = result_state[0]
        if end_epsilon_decaying >= episodes >= start_epsilon_decaying:
            epsilon = epsilon - epsilon_decay_value
        if current_state == "In":
            done = True

    i += 1

for state in states:
    index_of_state = states.index(state)
    min = np.min(q_table[index_of_state])
    index_optimal_action = q_table[index_of_state].index(min)
    optimal_action = actions[index_of_state][index_optimal_action]
    optimal_policy.append([state, optimal_action])

print("The optimal policy is:")
print(optimal_policy)

print(q_table)

