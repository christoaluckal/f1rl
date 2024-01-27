import numpy as np
import torch
import random

class EpsilonGreedy():
    def __init__(self) -> None:
        pass
    
    def policy(state, online_network,action_count,epsilon,device):

        rand = np.random.uniform(0,1)

        if rand > epsilon:
            action_list = online_network(torch.tensor(state).to(device))
            return [torch.argmax(action_list).item(),0]
        else:
            choice = random.choice(range(action_count))
            return [choice,1]