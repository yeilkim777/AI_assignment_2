import argparse
import random
from random import choices
import sys
import os
import copy
import numpy as np


# ----------GLOBAL VARS----------

states = []
actions = []
result_states = []
result_probabilities = []
predicted_probabilities = []

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
            predicted_probabilities.append([])
    for line in sections:
        parts = line.split('/')
        Index = states.index(parts[0])
        if parts[1] not in actions[Index]:
            actions[Index].append(parts[1])
            result_states[Index].append([])
            result_probabilities[Index].append([])
            predicted_probabilities[Index].append([])
            #print(Index,actions[Index].index(parts[1]))
        if parts[2] not in result_states[Index][actions[Index].index(parts[1])]:
            result_states[Index][actions[Index].index(parts[1])].append(parts[2])
            result_probabilities[Index][actions[Index].index(parts[1])].append(float(parts[3]))
            predicted_probabilities[Index][actions[Index].index(parts[1])].append(0)

states.append("In")
actions.append(["Putt"])
result_states.append([["In"]])
result_probabilities.append([[1]])
predicted_probabilities.append([[0]])

temp_table = copy.deepcopy(actions)
for i in range(len(temp_table)):
    for j in range(len(temp_table[i])):
        temp_table[i][j] = 0

utility_table = [temp_table]
probability_results =[]
optimal_policy = []


def main():

    episodes = 5000
    epsilon = 0.5 # to determine exploration/exploitation
    start_epsilon_decaying = 1
    end_epsilon_decaying = episodes / 2
    epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)
    discount = 0.9

    # predicting probability table
    for state in states:
        index_of_state = states.index(state)
        for action in actions[index_of_state]:
            index_of_action = actions[index_of_state].index(action)
            all_results = []
            i = 0
            close = 0
            same = 0
            left = 0
            ravine = 0
            over = 0
            inhole = 0
            while i < 5000: #by iterating 5000 times, to get random results, we can get probabilities for each resulting state.
                result = choices(result_states[index_of_state][index_of_action],
                                 result_probabilities[index_of_state][index_of_action]) #getting a random resulting state from all possible resulting states and probabilities.
                all_results.append(result[0]) # for each action, getting all returned resulting states
                i = i + 1
            for result in all_results: # according to all the resulting states, we can calculating how many times each one appears.
                if result == "Close":
                    close = close + 1
                elif result == "Same":
                    same = same + 1
                elif result == "Left":
                    left = left + 1
                elif result == "Ravine":
                    ravine = ravine + 1
                elif result == "Over":
                    over = over + 1
                elif result == "In":
                    inhole = inhole + 1
            for result in result_states[index_of_state][index_of_action]: # with the number of each resulting state appears, we can calculate the probabilities.
                if result == "Close":
                    predicted_probabilities[index_of_state][index_of_action][
                        result_states[index_of_state][index_of_action].index(result)]= close/ 5000
                elif result == "Same":
                    predicted_probabilities[index_of_state][index_of_action][
                        result_states[index_of_state][index_of_action].index(result)] = same / 5000
                elif result == "Left":
                    predicted_probabilities[index_of_state][index_of_action][
                        result_states[index_of_state][index_of_action].index(result)] = left / 5000
                elif result == "Ravine":
                    predicted_probabilities[index_of_state][index_of_action][
                        result_states[index_of_state][index_of_action].index(result)] = ravine / 5000
                elif result == "Over":
                    predicted_probabilities[index_of_state][index_of_action][
                        result_states[index_of_state][index_of_action].index(result)] = over / 5000
                elif result == "In":
                    predicted_probabilities[index_of_state][index_of_action][
                        result_states[index_of_state][index_of_action].index(result)] = inhole / 5000

    print("This part if the predicted resulting state probabilities. The first array is all the states, "
          "second array is all the actions for each state and the third array is for each action, "
          "what the resulting states are and the last array is the probability for each resulting state.")
    print(states)
    print(actions)
    print(result_states)
    print(predicted_probabilities)
    for state in states: # creating the starting state, action, and resulting state triplet
        index_of_state = states.index(state)
        for action in actions[index_of_state]:
            index_of_action = actions[index_of_state].index(action)
            for result_state in result_states[index_of_state][index_of_action]:
                index_of_result_state = result_states[index_of_state][index_of_action].index(result_state)
                probability = predicted_probabilities[index_of_state][index_of_action][index_of_result_state]
                triplet = [state, action, result_state, probability]
                probability_results.append(triplet)

    print("All triplets of starting state, action and resulting state:")
    print(probability_results)

    # calculating utility table

    i = 0
    while i < episodes:
        new_utility_array = [] # each iteration
        for state in states:
            index_of_state = states.index(state)
            new_state_utility = []
            for action in actions[index_of_state]:
                index_of_action = actions[index_of_state].index(action)
                future_value = 0
                for result_state in result_states[index_of_state][index_of_action]:
                    index_of_result_state = result_states[index_of_state][index_of_action].index(result_state)
                    index_in_utility = states.index(result_state)
                    if np.random.random() > epsilon: # calculating utility value from the resulting state, its probability and its corresponding utility,
                        # and when doing exploitation, we choose the minimum utility and its corresponding action
                        future_value = future_value + np.min(utility_table[i][index_in_utility]) * \
                                       predicted_probabilities[index_of_state][index_of_action][index_of_result_state]
                    else: # when doing exploration, we choose an utility of a random action from that state
                        future_value = future_value + random.choice(utility_table[i][index_in_utility]) * \
                                       predicted_probabilities[index_of_state][index_of_action][index_of_result_state]
                if state != "In":
                    new_utility = 1 + discount * future_value
                else:
                    new_utility = 0
                new_state_utility.append(new_utility)
            new_utility_array.append(new_state_utility)
        utility_table.append(new_utility_array)
        i = i + 1

        if end_epsilon_decaying >= episodes >= start_epsilon_decaying:
            epsilon = epsilon - epsilon_decay_value

    # Deciding optimal policy

    for state in states:
        index_of_state = states.index(state)
        min = np.min(utility_table[episodes][index_of_state])
        index_optimal_action = utility_table[episodes][index_of_state].index(min)
        optimal_action = actions[index_of_state][index_optimal_action]
        optimal_policy.append([state, optimal_action])

    print("The optimal policy is:")
    print(optimal_policy)


if __name__== "__main__":
  main()