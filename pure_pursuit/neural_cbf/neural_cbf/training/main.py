from argparse import ArgumentParser
from copy import copy

import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from neural_cbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)

from neural_cbf.neural_cbf import NeuralCBFController
from train_dynamics import DynamicsModel
from neural_cbf.systems import InvertedPendulum
from neural_cbf.models import Policy, CBFNet

def main(args):
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    controller_period = 0.05
    simulation_dt = 0.01
    
    simulator = InvertedPendulum(
        nominal_params,
    )
    
    initial_conditions = [
        (-np.pi / 2, np.pi / 2),  # theta
        (-1.0, 1.0),  # theta_dot
    ]
    
    data_module = EpisodicDataModule(
        simulator,
        initial_conditions,
        trajectories_per_episode=1,  # disable collecting data from trajectories
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
        quotas={"safe": 0.4, "unsafe": 0.2, "goal": 0.2},
    )
    
    
    # Initialize the controller
    dir_path = "/home/ny0221/neural_cbf/neural_cbf/training/checkpoints/"
    nn_dynamics = DynamicsModel.load_from_checkpoint(dir_path + "pendulum.ckpt", n_dims=simulator.n_dims, n_controls=simulator.n_controls, control_limits=simulator.control_limits)    

    policy_net = Policy(simulator.n_dims, simulator.n_controls)
    V = CBFNet(simulator.n_dims)
    
    neural_cbf_controller = NeuralCBFController(
        nn_dynamics,
        data_module,
        simulator,
        cbf_lambda=1.0,
        safe_level=0.0,
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=1000,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(neural_cbf_controller)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
