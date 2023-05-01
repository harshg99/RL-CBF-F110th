import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from argparse import ArgumentParser
from torch import Tensor, nn
from typing import Iterator, List, Tuple
from torch.optim import Adam, Optimizer
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
from policy import Policy
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from collections import OrderedDict
class CBFNet(nn.Module):
    """Simple MLP network for computing V(x)"""
    def __init__(self, n_dims: int, hidden_sizes: List[int] = [8]):
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
        V = 0.5*(V*V).sum(dim=1)
        return V



class NeuralCBFController(pl.LightningModule):
    """
    V(goal) = 0
    V >= 0
    V(safe) < c
    V(unsafe) > c
    dV/dt <= -lambda V
    """
    def __init__(self, dynamics_model, datamodule, cbf_lambda=1.0, safe_level=1.0):
        super().__init__()
        #self.save_hyperparameters()
        self.dynamics_model = dynamics_model
        n_dims = dynamics_model.n_dims
        n_controls = dynamics_model.n_controls
        self.policy_net = Policy(n_dims, n_controls)
        self.V = CBFNet(n_dims)
        # Save the datamodule
        self.datamodule = datamodule        
        self.unsafe_level = safe_level 
        self.safe_level = safe_level
        self.cbf_lambda = cbf_lambda
        
    def descent_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask:
        torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
        requires_grad: bool = False,
        ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to the descent condition

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
            requires_grad: if True, use a differentiable QP solver
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []
        eps = 0.1
        #TODO modify this value, and don't put it here
        a_3 = 10
        JV = torch.eye(self.V.net[0].in_features) 
        V = x
        for layer in self.V.net:
            V = layer(V)
            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)
        
        JV = torch.bmm(V.unsqueeze(1), JV).squeeze() 
        #jacobian(self.V, x)
        Lf_V = (JV * self.dynamics_model(x, self.policy_net(x))).sum(axis=1)
        descent_violation = F.relu(eps + Lf_V + self.cbf_lambda*V.squeeze())
        r = self.estimate_violation(batch)
        cbf_descent_term = a_3*descent_violation.mean()  
        loss.append(("CBF descent term", cbf_descent_term))
        return loss
    
    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        JV = torch.eye(self.V.net[0].in_features) 
        V = x
        for layer in self.V.net:
            V = layer(V)
            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V)), JV)
        
        JV = torch.bmm(V.unsqueeze(1), JV).squeeze() 
       
        return V, JV 
    def boundary_loss(
        self,
        x: torch.Tensor,
        goal_mask: torch.Tensor,
        safe_mask: torch.Tensor,
        unsafe_mask: torch.Tensor,
        accuracy: bool = False,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Evaluate the loss on the CLBF due to boundary conditions

        args:
            x: the points at which to evaluate the loss,
            goal_mask: the points in x marked as part of the goal
            safe_mask: the points in x marked safe
            unsafe_mask: the points in x marked unsafe
            accuracy: if True, return the accuracy (from 0 to 1) as well as the losses
        returns:
            loss: a list of tuples containing ("category_name", loss_value).
        """
        eps = 1e-2
        # Compute loss to encourage satisfaction of the following conditions...
        loss = []
        V = self.V(x)

        #   1.) CLBF should be minimized on the goal point
        goal_point = torch.zeros([1, self.dynamics_model.n_dims])
        V_goal_pt = self.V(goal_point)
        goal_term = 1e1 * V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        #   2.) 0 < V <= safe_level in the safe region
        V_safe = V[safe_mask]
        safe_violation = F.relu(eps + V_safe - self.safe_level)
        safe_V_term = 1e2 * safe_violation.mean()
        loss.append(("CLBF safe region term", safe_V_term))
        if accuracy:
            safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("CLBF safe region accuracy", safe_V_acc))

        #   3.) V >= unsafe_level in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
        unsafe_V_term = 1e2 * unsafe_violation.mean()
        loss.append(("CLBF unsafe region term", unsafe_V_term))
        if accuracy:
            unsafe_V_acc = (
                unsafe_violation <= eps
            ).sum() / unsafe_violation.nelement()
            loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss
        
        
    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask = batch

        # Compute the losses
        component_losses = {}
        component_losses.update(
            self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask)
        )
        component_losses.update(
            self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, requires_grad=True)
        )

        # Compute the overall loss by summing up the individual losses
        total_loss = torch.tensor(0.0).type_as(x)
        # For the objectives, we can just sum them
        for _, loss_value in component_losses.items():
            if not torch.isnan(loss_value):
                total_loss += loss_value

        batch_dict = {"loss": total_loss, **component_losses}

        return batch_dict
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.V.parameters(), lr=1e-3)
        return optimizer
    
    def prepare_data(self):
        return self.datamodule.prepare_data()

 
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def estimate_violation(self, batch, n_samples=100): 
        x, goal_mask, safe_mask, unsafe_mask = batch
        batch_size = x.shape[0] 
        n_controls = self.dynamics_model.n_controls
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_samples = torch.rand([batch_size * n_samples, n_controls])
        u_samples = u_samples*(upper_lim - lower_lim) + lower_lim
        V, JV = self.V_with_jacobian(x)
        f_dot = self.dynamics_model(x.repeat_interleave(n_samples, dim=0), u_samples).view(batch_size, n_samples, 5)
        lie_derivatives = torch.bmm(f_dot,JV.unsqueeze(2)).squeeze() 
        violation = (lie_derivatives + self.cbf_lambda*V).max(axis=1)[0]
        return F.relu(violation) 
        
