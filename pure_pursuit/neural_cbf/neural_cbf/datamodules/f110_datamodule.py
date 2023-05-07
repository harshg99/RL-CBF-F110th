
from typing import List, Callable, Tuple, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


from scripts.create_data import F110System, parse_args
import numpy as np
from pdb import set_trace as bp

class F110DataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        model: F110System,
        val_split: float = 0.1,
        batch_size: int = 512,
        quotas={"safe": 0.5, "unsafe": 0.3, "goal": 0.2},
    ):
        super().__init__()
        # Args that need to be parsed into from the training script
        self.args = args
        self.model = model(args)
        self.val_split = val_split
        self.batch_size = batch_size
        self.quotas = quotas


    def prepare_data(self):

        safe_data, unsafe_data, metadata = self.model.load_data()

        # Randomly split data into training and test sets
        
        safe_states = safe_data['states']
        unsafe_states = unsafe_data['states']
        safe_control = safe_data['controls']
        unsafe_control = unsafe_data['controls']

        start_point = metadata['start_point']
        goal_point = metadata['goal']
        
        near_safe_goals = np.linalg.norm(safe_states[:, :2] - goal_point, axis=1) < self.args.goal_radius
        near_unsafe_goals = np.linalg.norm(unsafe_states[:, :2] - goal_point, axis=1) < self.args.goal_radius

        safe_mask = np.expand_dims(np.concatenate((np.ones(safe_states.shape[0],dtype=np.int),
         np.zeros(unsafe_states.shape[0],dtype=np.int)),axis = 0), axis=-1)
        goal_mask = np.expand_dims(np.concatenate((near_safe_goals, near_unsafe_goals),axis = 0), axis=-1)
        states = np.concatenate((safe_states, unsafe_states), axis=0)
        control = np.concatenate((safe_control, unsafe_control), axis=0)

        random_indices = torch.randperm(states.shape[0])
        val_pts = int(states.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]
        
        self.states_training = states[training_indices]
        self.states_validation = states[validation_indices]
        self.control_training = control[training_indices]
        self.control_validation = control[validation_indices]
        self.safe_mask_training = safe_mask[training_indices]
        self.safe_mask_validation = safe_mask[validation_indices]
        self.goal_mask_training = goal_mask[training_indices]
        self.goal_mask_validation = goal_mask[validation_indices]

        #bp()
        print("Full dataset:")
        print(f"\t{self.states_training.shape[0]} training")
        print(f"\t{self.states_validation.shape[0]} validation")
        print("\t----------------------")
        print(f"\t{self.goal_mask_training.sum()} goal points")
        print(f"\t({self.goal_mask_validation.sum()} val)")
        print(f"\t{self.safe_mask_training.sum()} safe points")
        print(f"\t({self.safe_mask_validation.sum()} val)")
        print(f"\t{np.logical_not(self.safe_mask_training).sum()} unsafe points")
        print(f"\t({np.logical_not(self.safe_mask_validation).sum()} val)")

        # Turn these into tensor datasets
        bp()
        self.training_data = TensorDataset(
            self.states_training,
            self.goal_mask_training,
            self.safe_mask_training,
            np.logical_not(self.safe_mask_training),
            self.control_training
        )
        self.validation_data = TensorDataset(
            self.states_validation,
            self.goal_mask_validation,
            self.safe_mask_validation,
            np.logical_not(self.safe_mask_training),
            self.control_validation
        )

    def add_data(self):


        print("\nAdding data!\n")
        # Get some data points from simulations
        
        safe_data, unsafe_data, metadata = self.model.load_data()

        print(f"Sampled {self.args.num_samples} new points")

        safe_states = safe_data['states']
        unsafe_states = unsafe_data['states']
        safe_control = safe_data['controls']
        unsafe_control = unsafe_data['controls']
        start_point = metadata['start_point']
        goal_point = metadata['goal']
        
        near_safe_goals = np.linalg.norm(safe_states[:, :2] - goal_point, axis=1) < self.args.goal_radius
        near_unsafe_goals = np.linalg.norm(unsafe_states[:, :2] - goal_point, axis=1) < self.args.goal_radius

        safe_mask = np.expand_dims(np.concatenate((np.ones(safe_states.shape[0],dtype=np.int), 
        np.zeros(unsafe_states.shape[0],dtype=np.int)),axis = 0),axis=-1)
        goal_mask = np.expand_dims(np.concatenate((near_safe_goals, near_unsafe_goals),axis = 0),axis=-1)
        states = np.concatenate((safe_states, unsafe_states), axis=0)
        control = np.concatenate((safe_control, unsafe_control), axis=0)

        random_indices = torch.randperm(states.shape[0])
        val_pts = int(states.shape[0] * self.val_split)
        validation_indices = random_indices[:val_pts]
        training_indices = random_indices[val_pts:]

        states_training = states[training_indices]
        states_validation = states[validation_indices]
        control_training = control[training_indices]
        control_validation = control[validation_indices]
        safe_mask_training = safe_mask[training_indices]
        safe_mask_validation = safe_mask[validation_indices]
        goal_mask_training = goal_mask[training_indices]
        goal_mask_validation = goal_mask[validation_indices]

        self.states_training = torch.cat((self.states_training, states_training), dim=0)
        self.states_validation = torch.cat((self.states_validation, states_validation), dim=0)
        self.control_training = torch.cat((self.control_training, control_training), dim=0)
        self.control_validation = torch.cat((self.control_validation, control_validation), dim=0)
        self.safe_mask_training = torch.cat((self.safe_mask_training, safe_mask_training), dim=0)
        self.safe_mask_validation = torch.cat((self.safe_mask_validation, safe_mask_validation), dim=0)
        self.goal_mask_training = torch.cat((self.goal_mask_training, goal_mask_training), dim=0)
        self.goal_mask_validation = torch.cat((self.goal_mask_validation, goal_mask_validation), dim=0)

        print("Full dataset:")
        print(f"\t{self.states_training.shape[0]} training")
        print(f"\t{self.states_validation.shape[0]} validation")
        print("\t----------------------")
        print(f"\t{self.goal_mask_training.sum()} goal points")
        print(f"\t({self.goal_mask_validation.sum()} val)")
        print(f"\t{self.safe_mask_training.sum()} safe points")
        print(f"\t({self.safe_mask_validation.sum()} val)")
        print(f"\t{np.logical_not(self.safe_mask_training).sum()} unsafe points")
        print(f"\t({np.logical_not(self.safe_mask_validation).sum()} val)")

        # Turn these into tensor datasets
        self.training_data = TensorDataset(
            self.states_training,
            self.goal_mask_training,
            self.safe_mask_training,
            np.logical_not(self.safe_mask_training),
            self.control_training
        )
        self.validation_data = TensorDataset(
            self.states_validation,
            self.goal_mask_validation,
            self.safe_mask_validation,
            np.logical_not(self.safe_mask_training),
            self.control_validation
        )
    
    def train_dataloader(self):
        """Make the DataLoader for training data"""
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=4,
        )

if __name__=="__main__":
    parser = parse_args(False)
    parser.add_argument('--goal_radius', type=float, default=0.5)
    args = parser.parse_args()

    system = F110System
    data_module = F110DataModule(args,system)
    # Test module setup
    data_module.prepare_data()

    # Test add data
    data_module.add_data()
    