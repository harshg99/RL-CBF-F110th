from argparse import ArgumentParser
from copy import copy

import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from neural_cbf.neural_cbf.datamodules import F110DataModule

from neural_cbf.neural_cbf.training import NeuralCBFController, F110DynamicsModel

from scripts.create_data import F110System, parse_args
import numpy as np

def main(args):


    controller_period = 0.05
    simulation_dt = 0.01

    # State_dims
    state_dims = 5
    control_dims = 2

    system = F110System(args)

    dynamics = F110DynamicsModel(n_dims = state_dims, n_controls = control_dims)

    data_module = F110DataModule(args,
        model=F110System,
        val_split=0.1,
        batch_size=32,
        quotas={"safe": 0.5, "unsafe": 0.4, "goal": 0.1},
    )
    
    
    # Initialize the controller
    dir_path = "neural_cbf/training/checkpoints/"

    neural_cbf_controller = NeuralCBFController(
        dynamics,
        data_module,
        system = system,
        cbf_lambda=1.0,
        safe_level=0.0,
    )
    dir_path = "neural_cbf/training/logs/"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=dir_path)
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=1000,
        logger=tb_logger,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(neural_cbf_controller)
    
if __name__ == "__main__":
    parser = parse_args(False)
    parser.add_argument('--goal_radius', type=float, default=0.4)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
