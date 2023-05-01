from neural_cbf.systems import KSCar

import pytorch_lightning as pl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR

controller_period = 0.01
simulation_dt = 0.001

class DynamicsDataModule(pl.LightningDataModule):
    def __init__(self, dynamics_model, batch_size=32):
        super().__init__()
        self.simulator = dynamics_model
        self.batch_size = batch_size

    def setup(self, stage=None):
        states, actions, next_states = self.simulator.sample_dynamics_data(500000)
        dataset = TensorDataset(states, actions, next_states)
        self.train_data, self.val_data, self.test_data = random_split(dataset, [400000, 50000, 50000])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class DynamicsModel(pl.LightningModule):
    def __init__(self, n_dims, n_controls, control_limits):
        super(DynamicsModel, self).__init__()
        self.layer1 = torch.nn.Linear(n_dims + n_controls, 32)
        self.layer2 = torch.nn.Linear(32, 64)
        self.layer3 = torch.nn.Linear(64, n_dims)
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

    def training_step(self, batch, batch_idx):
        x, u, x_prime = batch
        x_prime_hat = self.forward(x, u)
        loss = self.loss_fn(x_prime_hat, x_prime)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, u, x_prime = batch
        x_prime_hat = self.forward(x, u)
        loss = self.loss_fn(x_prime_hat, x_prime)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, u, x_prime = batch
        x_prime_hat = self.forward(x, u)
        loss = self.loss_fn(x_prime_hat, x_prime)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
        return [optimizer], [lr_scheduler]


def main(args):
    # Define the dynamics model
    nominal_params = {
        "psi_ref": 1.0,
        "v_ref": 10.0,
        "a_ref": 0.0,
        "omega_ref": 0.0,
    }
    simulator = KSCar(
        nominal_params, dt=simulation_dt, controller_dt=controller_period
    )
    #states, actions, next_states = dynamics_model.sample_dynamics_data(10000)
    dynamics_model = DynamicsModel(simulator.n_dims, simulator.n_controls, simulator.control_limits)
    data_module = DynamicsDataModule(simulator, batch_size=32)
    trainer = pl.Trainer(max_epochs=5000, gpus=1)
    trainer.fit(dynamics_model, datamodule=data_module)
    #trainer.test(dynamics_model, datamodule=data_module)

    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    main(args)
