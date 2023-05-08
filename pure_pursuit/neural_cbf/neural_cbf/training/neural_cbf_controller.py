import pytorch_lightning as pl
import torch

from torch import Tensor, nn
from typing import Iterator, List, Tuple
from torch.optim import Adam, Optimizer

import torch.nn.functional as F

from neural_cbf.neural_cbf.models import Policy, CBFNet
from torch.optim.lr_scheduler import StepLR
import numpy as np
import math

def kaiming_init(model):
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif name.startswith("layers.0"):  # The first layer does not have ReLU applied on its input
            param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
        else:
            param.data.normal_(0, math.sqrt(2) / math.sqrt(param.shape[1]))

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
        self.policy_net = Policy(n_dims, n_controls, hidden_size=512)
        self.V = CBFNet(n_dims, hidden_sizes=[128,1028,1028,128])

        kaiming_init(self.V)
        kaiming_init(self.policy_net)

        # Save the datamodule
        self.datamodule = datamodule        
        self.unsafe_level = safe_level 
        self.safe_level = safe_level
        self.cbf_lambda = cbf_lambda
        
        self.policy_loss_weight = 2.0
        self.goal_loss_weight = 0.2
        self.safe_loss_weight = 2.0
        self.unsafe_loss_weight = 2.0
        self.descent_loss_weight = 4.0
        self.descent_loss_weight_policy = 2.0
        self.positive_value_loss_weight = 10.0

        self.system = system
        self.goal_point = system.controller.goal_point

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

        #No grdients passed through the policy for the descent on the CLBF
        U = self.policy_net(x)
        U = U.detach()
        (f,g) = self.dynamics_model(x.cpu().detach().numpy(), U.cpu().detach().numpy())
        full_dyn =  f.to(self.device) + torch.bmm(g.to(self.device), U.unsqueeze(2)).squeeze().to(self.device)
        Lf_V = (JV * full_dyn).sum(axis=1)
        descent_violation = F.relu(eps + Lf_V + self.cbf_lambda*V.squeeze())
        self.log("train/descent_violation", descent_violation.mean(), logger=True)
        #estimated_r = self.estimate_violation(x)
        cbf_descent_term = self.descent_loss_weight*descent_violation.mean()  
        loss.append(("CBF descent term", cbf_descent_term))
        return loss
    
    def V_with_jacobian(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        JV = torch.eye(self.V.net[0].in_features).to(self.device)
        V = x
        for layer in self.V.net:
            V = layer(V)
            if isinstance(layer, nn.Linear):
                JV = torch.matmul(layer.weight, JV)
            elif isinstance(layer, nn.ReLU):
                JV = torch.matmul(torch.diag_embed(torch.sign(V.detach())), JV)
        

        JV = torch.bmm(V.unsqueeze(1), JV).squeeze() 
        V = 0.5 * (V * V)
       
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
        # TODO: just added this
        V = 0.5 * (V * V)

        self.log("train/V", V.mean(), logger=True)

        # 0. V should be positive definite
        # positive_value_loss = self.positive_value_loss_weight*F.relu(-V).mean()

        #   1.) CLBF should be minimized on the goal point
        #goal_point = torch.zeros([1, self.dynamics_model.n_dims])
        # Samplling random 1000 goal points with set poses but random steering angle, velocity and acceleration
        goal_points = torch.zeros([500, self.dynamics_model.n_dims]).to(self.device)

        # Population position
        goal_points[:,0] = self.goal_point[0]
        goal_points[:,1] = self.goal_point[1]

        #Pooulating steering angle, velocity and acceleration
        limits = torch.tensor([self.system.args.steering_max*2, self.system.args.vel_upper, 2*np.pi]).to(self.device)
        dev = torch.tensor([self.system.args.steering_max, 0, np.pi]).to(self.device)
        goal_points[:,2:] = torch.rand(500, self.dynamics_model.n_dims-2).to(self.device) * limits - dev
        #V_goal_pt = torch.square(self.V(goal_points))
        V_goal_pt = 0.5 * self.V(goal_points) * self.V(goal_points)
        goal_term = self.goal_loss_weight* V_goal_pt.mean()

        # goals in the data buffer setting those to zero
        #V_goal = torch.square(V[goal_mask])
        V_goal = V[goal_mask]
        goal_term += V_goal.mean()

        loss.append(("CLBF goal term", goal_term))

        #   2.) 0 < V <= safe_level in the safe region
        V_safe = V[safe_mask]
        safe_violation = F.relu(eps + V_safe - self.safe_level)
        safe_V_term = self.safe_loss_weight * safe_violation.mean()
        loss.append(("CLBF safe region term", safe_V_term))
        if accuracy:
            safe_V_acc = (safe_violation <= eps).sum() / safe_violation.nelement()
            loss.append(("CLBF safe region accuracy", safe_V_acc))

        self.log("train/safe_violation", safe_violation.mean(),prog_bar=True, logger=True)
        #   3.) V >= unsafe_level in the unsafe region
        V_unsafe = V[unsafe_mask]
        unsafe_violation = F.relu(eps + self.unsafe_level - V_unsafe)
        unsafe_V_term = self.unsafe_loss_weight * unsafe_violation.mean()
        loss.append(("CLBF unsafe region term", unsafe_V_term))
        self.log("train/unsafe_violation", unsafe_violation.mean(),prog_bar=True, logger=True)
        if accuracy:
            unsafe_V_acc = (
                unsafe_violation <= eps
            ).sum() / unsafe_violation.nelement()
            loss.append(("CLBF unsafe region accuracy", unsafe_V_acc))

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        # print(optimizer_idx)
        x, goal_mask, safe_mask, unsafe_mask, control = batch
        goal_mask = goal_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        unsafe_mask = unsafe_mask.bool().squeeze()

        if optimizer_idx == 0:
            # Compute the losses
            boundary_losses = {}
            boundary_losses.update(self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask))


            descent_losses = {}
            descent_losses.update(self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, requires_grad=True))

            # Compute the overall loss by summing up the individual losses
            violation_loss = torch.tensor(0.0).type_as(x)

            boundary_loss = torch.tensor(0.0).type_as(x)

            for loss_name, loss_value in boundary_losses.items():
                if not torch.isnan(loss_value):
                    boundary_loss += loss_value
                    self.log("train/" + loss_name, loss_value)

            descent_loss = torch.tensor(0.0).type_as(x)
            for loss_name, loss_value in descent_losses.items():
                if not torch.isnan(loss_value):
                    descent_loss += loss_value
                    self.log("train/"+ loss_name, loss_value)
                    if 'descent_violation' in loss_name:
                        violation_loss += loss_value

            if torch.isnan(boundary_loss):
                value_loss = descent_loss
            elif torch.isnan(descent_loss):
                value_loss = boundary_loss
            else:
                value_loss = boundary_loss + descent_loss

            self.log("train/CLBF_loss", value_loss.cpu().detach().numpy().item(), logger=True)
            self.log("train/descent_loss", descent_loss.cpu().detach().numpy().item(), prog_bar=True, logger=True)
            self.log("train/boundary_loss", boundary_loss.cpu().detach().numpy().item(), logger=True)
            return value_loss
        elif optimizer_idx == 1:
            # For the objectives, we can just sum them
            # batch_dict = {"loss": total_loss, **component_losses}

            # Update the CBF here


            policy_loss = torch.nn.MSELoss()(self.policy_net(x), control)
            total_loss = self.policy_loss_weight * policy_loss

            eps = 0.1
            V, JV = self.V_with_jacobian(x)

            # No gradients passed through value network for the descent loss on the policy
            U = self.policy_net(x)
            V = V.detach()
            JV = JV.detach()
            (f, g) = self.dynamics_model(x.cpu().detach().numpy(), U.cpu().detach().numpy())
            full_dyn = f.to(self.device) + torch.bmm(g.to(self.device), U.unsqueeze(2)).squeeze().to(self.device)
            Lf_V = (JV * full_dyn).sum(axis=1)
            descent_violation = F.relu(eps + Lf_V + self.cbf_lambda * V.squeeze())
            descent_loss_policy = self.descent_loss_weight_policy * descent_violation.mean()
            policy_loss += descent_loss_policy

            total_loss += policy_loss
            violation_loss = descent_violation.mean()

            # self.log("train/train_loss", total_loss.cpu().detach().numpy().item(), on_step=True, prog_bar=True,
            #          logger=True)
            self.log("train/policy_loss", policy_loss.cpu().detach().numpy().item(), logger=True)

            self.log("train/descent_loss_policy", descent_loss_policy.cpu().detach().numpy().item(), prog_bar=True, logger=True)
            self.log("train/violation_loss", violation_loss.cpu().detach().numpy().item(), prog_bar=True,logger=True)

            return total_loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        V_optimizer = Adam(self.V.parameters(), lr=1e-4)
        policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-4)
        lr_scheduler = StepLR(V_optimizer, step_size=100, gamma=0.99)
        policy_scheduler = StepLR(policy_optimizer,step_size = 100,gamma = 0.99)
        return [V_optimizer, policy_optimizer], [lr_scheduler,policy_scheduler]

    
    def prepare_data(self):
        return self.datamodule.prepare_data()

 
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def validation_step(self, batch, batch_idx):
        '''Conduct the validation step for the given batch'''
        # Extract the input and masks from the batch
        x, goal_mask, safe_mask, unsafe_mask,control = batch
        goal_mask = goal_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        unsafe_mask = unsafe_mask.bool().squeeze()

        # Compute the losses
        boundary_losses = {}
        boundary_losses.update(self.boundary_loss(x, goal_mask, safe_mask, unsafe_mask))

        descent_losses = {}
        descent_losses.update(self.descent_loss(x, goal_mask, safe_mask, unsafe_mask, requires_grad=True))

        # Compute the overall loss by summing up the individual losses
        # Compute the overall loss by summing up the individual losses
        violation_loss = torch.tensor(0.0).type_as(x)

        boundary_loss = torch.tensor(0.0).type_as(x)

        for loss_name, loss_value in boundary_losses.items():
            if not torch.isnan(loss_value):
                boundary_loss += loss_value
                self.log("val/" + loss_name, loss_value)
                if 'violation' in loss_name:
                    violation_loss += loss_value

        descent_loss = torch.tensor(0.0).type_as(x)
        for loss_name, loss_value in descent_losses.items():
            if not torch.isnan(loss_value):
                descent_loss += loss_value
                self.log("val/" + loss_name, loss_value)
                if 'violation' in loss_name:
                    violation_loss += loss_value

        if torch.isnan(boundary_loss):
            value_loss = descent_loss
        elif torch.isnan(descent_loss):
            value_loss = boundary_loss
        else:
            value_loss = boundary_loss + descent_loss
        # For the objectives, we can just sum them
        # batch_dict = {"loss": total_loss, **component_losses}
        total_loss = value_loss
        policy_loss = torch.nn.MSELoss()(self.policy_net(x), control)
        total_loss += self.policy_loss_weight * policy_loss

        eps = 0.1
        V, JV = self.V_with_jacobian(x)

        # Detaching descent loss here
        U = self.policy_net(x)
        V = V.detach()
        JV = JV.detach()
        (f, g) = self.dynamics_model(x.cpu().detach().numpy(), U.cpu().detach().numpy())
        full_dyn = f.to(self.device) + torch.bmm(g.to(self.device), U.unsqueeze(2)).squeeze().to(self.device)
        Lf_V = (JV * full_dyn).sum(axis=1)
        descent_violation = F.relu(eps + Lf_V + self.cbf_lambda * V.squeeze())
        descent_loss_policy = self.descent_loss_weight * descent_violation.mean()
        policy_loss += descent_loss_policy

        violation_loss += descent_violation.mean()
        total_loss += policy_loss

        self.log("val/train_loss", total_loss.cpu().detach().numpy().item(),
                 logger=True)
        self.log("val/policy_loss", policy_loss.cpu().detach().numpy().item(),
                 logger=True)
        self.log("val/CLBF_loss", value_loss.cpu().detach().numpy().item(),
                 logger=True)
        self.log("val/descent_loss", descent_loss.cpu().detach().numpy().item(),
                 logger=True)
        self.log("val/boundary_loss", boundary_loss.cpu().detach().numpy().item(),
                 logger=True)
        self.log("val/descent_loss_policy", descent_loss_policy.cpu().detach().numpy().item(), logger=True)
        self.log("val/violation_loss", violation_loss.cpu().detach().numpy().item(), logger=True)
    # def estimate_violation(self, x, n_samples=100):
    #     batch_size = x.shape[0]
    #     n_controls = self.dynamics_model.n_controls
    #     n_dims = self.dynamics_model.n_dims
    #     upper_lim, lower_lim = self.dynamics_model.control_limits
    #     u_samples = torch.rand([batch_size * n_samples, n_controls])
    #     u_samples = u_samples*(upper_lim - lower_lim) + lower_lim
    #     V, JV = self.V_with_jacobian(x)
    #     f_dot,g_dot = self.dynamics_model(x.repeat_interleave(n_samples, dim=0), u_samples).view(batch_size, n_samples, n_dims)
    #     lie_derivatives = torch.bmm(f_dot,JV.unsqueeze(2)).squeeze()
    #     violation = (lie_derivatives + self.cbf_lambda*V).max(axis=1)[0]
    #     return F.relu(violation)
