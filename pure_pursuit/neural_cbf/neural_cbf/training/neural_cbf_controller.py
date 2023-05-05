import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from argparse import ArgumentParser
from torch import Tensor, nn
from typing import Iterator, List, Tuple
from torch.optim import Adam, Optimizer
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from collections import OrderedDict
from neural_cbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_cbf.models import Policy, CBFNet
from torch.optim.lr_scheduler import StepLR
class NeuralCBFController(pl.LightningModule):
    """
    V(goal) = 0
    V >= 0
    V(safe) < c
    V(unsafe) > c
    dV/dt <= -lambda V
    """
    def __init__(self, dynamics_model, datamodule, system, cbf_lambda=1.0, safe_level=0.0):
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
        
        self.policy_loss_weight = 0
        self.goal_loss_weight = 0
        self.safe_loss_weight = 10
        self.unsafe_loss_weight = 10 
        self.descent_loss_weight = 100 
        
        # TODO need to remove later
        self.dynamic_system = system 
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
        V, JV = self.V_with_jacobian(x)

        #Detaching descent loss here
        U = self.policy_net(x)
        U.requires_grad = False
        Lf_V = (JV * self.dynamics_model(x, U)).sum(axis=1)
        descent_violation = F.relu(eps + Lf_V + self.cbf_lambda*V.squeeze())

        #estimated_r = self.estimate_violation(x)
        cbf_descent_term = self.descent_loss_weight*descent_violation.mean()  
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
        #V = 0.5*(V*V)
       
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
        goal_term = self.goal_loss_weight* V_goal_pt.mean()
        loss.append(("CLBF goal term", goal_term))

        #   2.) 0 < V <= safe_level in the safe region
        V_safe = V[safe_mask]
        safe_violation = F.relu(eps + V_safe - self.safe_level)
        safe_V_term = self.safe_loss_weight * safe_violation.mean()
        loss.append(("CLBF safe region term", safe_V_term))
        if accuracy:
            safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("CLBF safe region accuracy", safe_V_acc))

        #   3.) V >= unsafe_level in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
        unsafe_V_term = self.unsafe_loss_weight * unsafe_violation.mean()
        loss.append(("CLBF unsafe region term", unsafe_V_term))
        if accuracy:
            unsafe_V_acc = (
                unsafe_violation <= eps
            ).sum() / unsafe_violation.nelement()
            loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss
        
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        print(optimizer_idx)
        x, goal_mask, safe_mask, unsafe_mask, control = batch

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

        #batch_dict = {"loss": total_loss, **component_losses}
        policy_loss = torch.nn.MSELoss()(self.policy_net(states[safe_mask]), control[safe_mask])  
        
        eps = 0.1
        V, JV = self.V_with_jacobian(x)

        #Detaching descent loss here
        U = self.policy_net(x)
        V.requires_grad = False
        JV.requires_grad = False
        Lf_V = (JV * self.dynamics_model(x, U)).sum(axis=1)
        descent_violation = F.relu(eps + Lf_V + self.cbf_lambda*V.squeeze())

        total_loss +=(self.policy_loss_weight * policy_loss + self.descent_loss_weight*descent_violation.mean())
        return total_loss
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        V_optimizer = Adam(self.V.parameters(), lr=1e-3)
        policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-3)      
        lr_scheduler = StepLR(V_optimizer, step_size=100, gamma=0.99)
        return [V_optimizer, policy_optimizer], [lr_scheduler]
    
        return optimizer
    
    def prepare_data(self):
        return self.datamodule.prepare_data()

 
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def estimate_violation(self, x, n_samples=100): 
        batch_size = x.shape[0] 
        n_controls = self.dynamics_model.n_controls
        n_dims = self.dynamics_model.n_dims
        upper_lim, lower_lim = self.dynamics_model.control_limits
        u_samples = torch.rand([batch_size * n_samples, n_controls])
        u_samples = u_samples*(upper_lim - lower_lim) + lower_lim
        V, JV = self.V_with_jacobian(x)
        f_dot = self.dynamics_model(x.repeat_interleave(n_samples, dim=0), u_samples).view(batch_size, n_samples, n_dims)
        lie_derivatives = torch.bmm(f_dot,JV.unsqueeze(2)).squeeze() 
        violation = (lie_derivatives + self.cbf_lambda*V).max(axis=1)[0]
        return F.relu(violation) 