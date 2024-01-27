import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

class NaiveDQN():
    def __init__(self) -> None:
        pass

    def train(self,online_network, target_network, memory, discount_rate,BUFFER_SAMPLE, rate,device):
        if len(memory) >= BUFFER_SAMPLE:

            chosen_experiences = random.sample(memory,BUFFER_SAMPLE)

            curr_states = [] # Current state values
            next_states = [] # Next state on action
            targets_batch = [] # Targets
            for i in range(len(chosen_experiences)):
                curr_states.append(chosen_experiences[i][0])
                next_states.append(chosen_experiences[i][3])

            curr_states = np.array(curr_states)
            next_states = np.array(next_states)
            
            curr_states = torch.tensor(curr_states).to(device)
            next_states = torch.tensor(next_states).to(device)

            Qs_table = online_network(curr_states)
            Qsd_table = target_network(next_states)
            target = Qs_table
            for i in range(len(chosen_experiences)):
                state,action,reward,new_state,terminated = chosen_experiences[i]
                if terminated:
                    Qsa = reward
                else:
                    Qsa = reward + discount_rate*torch.max(Qsd_table[i])
                t_q_value = Qs_table[i].cpu().detach().numpy()    
                t_q_value[action] = Qsa
                targets_batch.append(t_q_value)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(online_network.parameters(), lr=rate)
            optimizer.zero_grad()   # zero the gradient buffers
            loss = criterion(Qs_table,  torch.tensor(np.array(targets_batch)).to(device))
            loss.backward()
            optimizer.step()

        return online_network
    
class DoubleDQN():
    def __init__(self) -> None:
        pass

    def train(self,online_network, target_network, memory, discount_rate,BUFFER_SAMPLE, rate,device):
        if len(memory) >= BUFFER_SAMPLE:

            chosen_experiences = random.sample(memory,BUFFER_SAMPLE)

            curr_states = [] # Current state values
            next_states = [] # Next state on action
            targets_batch = [] # Targets
            for i in range(len(chosen_experiences)):
                curr_states.append(chosen_experiences[i][0])
                next_states.append(chosen_experiences[i][3])

            
            
            curr_states = torch.tensor(curr_states).to(device)
            next_states = torch.tensor(next_states).to(device)

            Qs_table_1 = online_network(curr_states)
            Qs_table_2 = online_network(next_states)
            Qsd_table = target_network(next_states)

            for i in range(len(chosen_experiences)):
                state,action,reward,new_state,terminated = chosen_experiences[i]
                action = torch.argmax(Qs_table_2[i])
                value = Qsd_table[i][action]
                if terminated:
                    Qsa = reward
                else:
                    Qsa = reward + discount_rate*value
                t_q_value = Qs_table_1[i].cpu().detach().numpy()    
                t_q_value[action] = Qsa
                targets_batch.append(t_q_value)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(online_network.parameters(), lr=rate)
            optimizer.zero_grad()   # zero the gradient buffers
            loss = criterion(Qs_table_1,  torch.tensor(np.array(targets_batch)).to(device))
            loss.backward()
            optimizer.step()

        return online_network