
from typing import List, Callable, Tuple, Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


from scripts.create_data import F110System, parse_args
import numpy as np
from pdb import set_trace as bp


from torch.utils.data import Dataset, DataLoader, Sampler
 
class F110Dataset(Dataset):
    def __init__(self, states, safe_mask, unsafe_mask, goal_mask, control, quotas=None):
        # convert into PyTorch tensors and remember them
        self.states = torch.tensor(states, dtype=torch.float32)
        self.safe_mask = torch.tensor(safe_mask, dtype=torch.float32)
        self.unsafe_mask = torch.tensor(unsafe_mask, dtype=torch.float32)
        self.goal_mask = torch.tensor(goal_mask, dtype=torch.float32)
        self.control = torch.tensor(control, dtype=torch.float32)

        self.num_safe_samples = self.safe_mask.sum()
        self.num_unsafe_samples = self.unsafe_mask.sum()
        self.num_goal_samples = (self.safe_mask * self.goal_mask).sum()

        self.safe_indices = torch.where(self.safe_mask == 1)[0]
        self.unsafe_indices = torch.where(self.unsafe_mask == 1)[0]
        self.goal_indices = torch.where(self.goal_mask*self.safe_mask == 1)[0]

        if quotas is None:
            quotas = {"safe": 0.5, "unsafe": 0.3, "goal": 0.2}
        self.quotas = quotas


    def __len__(self):
        # this should return the size of the dataset
        return self.states.shape[0]
 
    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.states[idx]
        safe_mask = self.safe_mask[idx]
        unsafe_mask = self.unsafe_mask[idx]
        goal_mask = self.goal_mask[idx]
        control = self.control[idx]
        return features, safe_mask, unsafe_mask, goal_mask, control

class F110StratifiedSampler(Sampler[int]):
    def __init__(self, data_source: F110Dataset, batch_size = 64, replacement: bool = False, seed = None) -> None:
        self.data_source = data_source
        self.quotas = data_source.quotas

        self.num_samples = len(self.data_source)

        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = int(seed)
        
        self.replacement = replacement

        generator = torch.Generator()
        generator.manual_seed(seed)
        self.generator = generator

        # Create a list of indices for each class such that the number of samples
        # from each class is proportional to the quota
        self.safe_proportion = (self.data_source.num_safe_samples - self.data_source.num_goal_samples) / self.num_samples
        self.unsafe_proportion = self.data_source.num_unsafe_samples / self.num_samples
        self.goal_proportion = self.data_source.num_goal_samples / self.num_samples

        weights_safe = self.quotas["safe"] / self.safe_proportion
        weights_unsafe = self.quotas["unsafe"] / self.unsafe_proportion
        weights_goal = self.quotas["goal"] / self.goal_proportion
        
        # All safe goal samples would be upweighted, unsafe fo
        self.weights = torch.zeros(self.num_samples)
        self.weights[self.data_source.safe_indices] = weights_safe
        self.weights[self.data_source.unsafe_indices] = weights_unsafe
        self.weights[self.data_source.goal_indices] = weights_goal
        self.batch_size = batch_size

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement,
         generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


class F110DataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        model: F110System,
        val_split: float = 0.1,
        batch_size: int = 512,
        quotas={"safe": 0.5, "unsafe": 0.4, "goal": 0.1},
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
        self.training_data = F110Dataset(
            self.states_training,
            self.safe_mask_training,
            np.logical_not(self.safe_mask_training),
            self.goal_mask_training,
            self.control_training
        )
        self.validation_data = F110Dataset(
            self.states_validation,
            self.safe_mask_validation,
            np.logical_not(self.safe_mask_validation),
            self.goal_mask_validation,
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

        self.states_training = np.concatenate((self.states_training, states_training), axis=0)
        self.states_validation = np.concatenate((self.states_validation, states_validation), axis=0)
        self.control_training = np.concatenate((self.control_training, control_training), axis=0)
        self.control_validation = np.concatenate((self.control_validation, control_validation), axis=0)
        self.safe_mask_training = np.concatenate((self.safe_mask_training, safe_mask_training), axis=0)
        self.safe_mask_validation = np.concatenate((self.safe_mask_validation, safe_mask_validation), axis=0)
        self.goal_mask_training = np.concatenate((self.goal_mask_training, goal_mask_training), axis=0)
        self.goal_mask_validation = np.concatenate((self.goal_mask_validation, goal_mask_validation), axis=0)

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
        self.training_data = F110Dataset(
            self.states_training,
            self.safe_mask_training,
            np.logical_not(self.safe_mask_training),
            self.goal_mask_training,
            self.control_training
        )
        self.validation_data = F110Dataset(
            self.states_validation,
            self.safe_mask_validation,
            np.logical_not(self.safe_mask_validation),
            self.goal_mask_validation,
            self.control_validation
        )
    
    def train_dataloader(self):
        """Make the DataLoader for training data"""
        sampler_train = F110StratifiedSampler(data_source=self.training_data, 
                                        replacement=True,
                                        seed = 0)

    
        return DataLoader(
            self.training_data,
            batch_size=self.batch_size,
            sampler=sampler_train,
            num_workers=4
        )

    def val_dataloader(self):
        """Make the DataLoader for validation data"""
        sampler_val = F110StratifiedSampler(data_source=self.validation_data,
                                replacement=True,
                                seed = 0)
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            sampler=sampler_val,
            num_workers=4
        )

if __name__=="__main__":
    parser = parse_args(False)
    parser.add_argument('--goal_radius', type=float, default=0.5)
    args = parser.parse_args()

    system = F110System
    data_module = F110DataModule(args,system)
    # Test module setup
    data_module.prepare_data()

    loader1 = data_module.train_dataloader()
    loader2 = data_module.val_dataloader()
    print(len(loader1))
    print(len(loader2))

    # Test add data
    data_module.add_data()
    