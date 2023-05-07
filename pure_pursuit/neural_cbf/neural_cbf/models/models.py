from torch import Tensor, nn
from collections import OrderedDict
from typing import Iterator, List, Tuple
import torch
class DynamicsModel(nn.Module):
    def __init__(self, n_dims, n_controls, control_limits):
        super(DynamicsModel, self).__init__()
        self.layer1 = torch.nn.Linear(n_dims + n_controls, 16)
        self.layer2 = torch.nn.Linear(16, 32)
        self.layer3 = torch.nn.Linear(32, n_dims)
        self.loss_fn = torch.nn.MSELoss()
        self.n_dims = n_dims
        self.n_controls = n_controls       
        self.control_limits = control_limits 
    def forward(self, x, u):
        x = torch.cat([x, u], dim=1)
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        x_prime = self.layer3(x)
        return x_prime
        
class Policy(nn.Module):
    """Simple MLP network."""
    def __init__(self, n_dims: int, n_actions: int, hidden_size: int = 32):
        """
        Args:
            obs_size: observation/state size of the environment
            n_actions: number of discrete actions available in the environment
            hidden_size: size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())

class CBFNet(nn.Module):
    """Simple MLP network for computing V(x)"""
    def __init__(self, n_dims: int, hidden_sizes: List[int] = [8, 16]):
        """
        Args:
            n_dims: input dimensionality
            hidden_sizes: size of hidden layers
        """ 
        super().__init__()
        self.layers: OrderedDict[str, nn.Module] = OrderedDict()
        self.layers["input_linear"] = nn.Linear(
            n_dims, hidden_sizes[0]
        )
        self.layers["input_activation"] = nn.ReLU()
        
        num_hidden_layer = len(hidden_sizes)
        
        for i in range(num_hidden_layer-1):
            self.layers[f"layer_{i}_linear"] = nn.Linear(
                hidden_sizes[i], hidden_sizes[i+1]
            )
            if i < num_hidden_layer - 1:
                self.layers[f"layer_{i}_activation"] = nn.ReLU()
        
        self.layers["output_linear"] = nn.Linear(hidden_sizes[-1], 1)
        
        self.net = nn.Sequential(self.layers)
    
    
    def forward(self, x):
        V = self.net(x)
        #V = 0.5*(V*V).sum(dim=1)
        return V
