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


class ILController(pl.LightningModule):
    """
    V(goal) = 0
    V >= 0
    V(safe) < c
    V(unsafe) > c
    dV/dt <= -lambda V
    """

    def __init__(self, dynamics_model, datamodule, system, cbf_lambda=1.0, safe_level=0.0):
        super().__init__()
        # self.save_hyperparameters()
        self.dynamics_model = dynamics_model
        self.system = system
        self.goal_point = system.controller.goal_point

        n_dims = dynamics_model.n_dims
        n_controls = dynamics_model.n_controls
        limits = np.array([[0, self.system.args.vel_upper],
                           [-self.system.args.steering_max, self.system.args.steering_max]])

        self.policy_net = Policy(n_dims, n_controls, hidden_size=512, limits=limits)

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


    def training_step(self, batch, batch_idx):
        """Conduct the training step for the given batch"""
        # Extract the input and masks from the batch
        # print(optimizer_idx)
        x, goal_mask, safe_mask, unsafe_mask, control = batch
        goal_mask = goal_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        unsafe_mask = unsafe_mask.bool().squeeze()

        policy_loss = torch.nn.MSELoss()(self.policy_net(x), control)
        total_loss = self.policy_loss_weight * policy_loss

        # self.log("train/train_loss", total_loss.cpu().detach().numpy().item(), on_step=True, prog_bar=True,
        #          logger=True)
        self.log("train/policy_loss", policy_loss.cpu().detach().numpy().item(), logger=True)

        return total_loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        policy_optimizer = Adam(self.policy_net.parameters(), lr=1e-4)
        policy_scheduler = StepLR(policy_optimizer, step_size=100, gamma=0.99)
        return [policy_optimizer], [policy_scheduler]

    def prepare_data(self):
        return self.datamodule.prepare_data()

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def validation_step(self, batch, batch_idx):
        '''Conduct the validation step for the given batch'''

        x, goal_mask, safe_mask, unsafe_mask, control = batch
        goal_mask = goal_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        safe_mask = safe_mask.bool().squeeze()
        unsafe_mask = unsafe_mask.bool().squeeze()

        policy_loss = torch.nn.MSELoss()(self.policy_net(x), control)
        total_loss = self.policy_loss_weight * policy_loss

        # self.log("train/train_loss", total_loss.cpu().detach().numpy().item(), on_step=True, prog_bar=True,
        #          logger=True)
        self.log("train/policy_loss", policy_loss.cpu().detach().numpy().item(), logger=True)

        return total_loss

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
