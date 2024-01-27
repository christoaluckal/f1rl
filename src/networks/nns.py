import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class Network_1(nn.Module):
    def __init__(self,states,actions,hidden) -> None:
        super().__init__()
        self.input = states
        self.hidden_nodes = hidden 
        self.output = actions
        print("Created Network 1")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes//2),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes//2, self.hidden_nodes//4),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes//4, self.output)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
class Network_2(nn.Module):
    def __init__(self,states,actions,hidden) -> None:
        super().__init__()
        self.input = states
        self.hidden_nodes = hidden 
        self.output = actions
        print("Created Network 2")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    

class Network_3(nn.Module):
    def __init__(self,states,actions,hidden) -> None:
        super().__init__()
        self.input = states
        self.hidden_nodes = hidden 
        self.output = actions
        print("Created Network 3")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes*2),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes*2, self.hidden_nodes*4),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes*4, self.output)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    

class Network_4(nn.Module):
    def __init__(self,states,actions,hidden) -> None:
        super().__init__()
        self.input = states
        self.hidden_nodes = hidden 
        self.output = actions
        print("Created Network 4")
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    

class General_Network(nn.Module):
    def __init__(self,n_states,n_actions,layer_list) -> None:
        super().__init__()
        self.state_count = n_states
        self.action_count = n_actions

        module_list = [nn.Linear(self.state_count,layer_list[0]).double(),nn.ReLU()]

        for i in range(1,len(layer_list)):
            module_list.append(nn.Linear(layer_list[i-1],layer_list[i]).double())
            module_list.append(nn.ReLU())

        module_list.append(nn.Linear(layer_list[-1],self.action_count).double())
        self.stack = nn.Sequential(*module_list)

        
    def forward(self,x):
        return self.stack(x)