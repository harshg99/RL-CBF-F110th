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

from neural_cbf_controller import NeuralCBFController
from train_dynamics import DynamicsModel
from neural_cbf.systems import KSCar

controller_period = 0.01
simulation_dt = 0.001


def main(args):
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    simulator = KSCar(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )

    # Initialize the DataModule
    initial_conditions = [
        (-0.1, 0.1),  # sxe
        (-0.1, 0.1),  # sye
        (-0.1, 0.1),  # delta
        (-0.1, 0.1),  # ve
        (-0.1, 0.1),  # psi_e
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
    
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    car_model = KSCar(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )
    
    # Initialize the controller
    dir_path = "/home/ny0221/neural_cbf/neural_cbf/training/checkpoints/"
    nn_dynamics = DynamicsModel.load_from_checkpoint(dir_path + "model.ckpt", n_dims=simulator.n_dims, n_controls=simulator.n_controls, control_limits=simulator.control_limits)    

    # Define the dynamics model
    neural_cbf_controller = NeuralCBFController(
        dynamics_model,
        data_module,
        cbf_lambda=1.0,
        safe_level=1.0,
    )
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=26,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(neural_cbf_controller)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
