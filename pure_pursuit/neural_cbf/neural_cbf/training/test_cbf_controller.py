import matplotlib.pyplot as plt
from neural_cbf.systems import InvertedPendulum
from matplotlib.pyplot import figure
import pandas as pd
import torch
import seaborn as sns
from train_dynamics import DynamicsModel
from neural_cbf.neural_cbf import NeuralCBFController
from neural_cbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
import numpy as np
from cbf_grid_verification import CBFVerificationExperiment

plot_x_label = '$\\theta$'
plot_y_label = '$\\dot{\\theta}$'

plot_x_nn_label = '$\\theta$_nn'
plot_y_nn_label = '$\\dot{\\theta}$_nn'

plot_x_index = 0
plot_y_index = 1
def run(x_start, params, system, controller, nn_dynamics):
    num_timesteps = 1000 
    delta_t = 0.01 
    x_current = x_start
    x_current_nn = x_start.clone() 
    n_sims = x_start.shape[0]
    results = []
    for tstep in range(num_timesteps):
        # Get the control input at the current state if it's time
        if tstep % 5 == 0:
            #u_current = controller.u(x_current)
            u_current = controller.policy_net(x_current).detach()
            u_current_nn = controller.policy_net(x_current_nn).detach()
        # Log the current state and control for each simulation
        for sim_index in range(n_sims):
            log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}

            # Include the parameters
            param_string = ""
            for param_name, param_value in params.items():
                param_value_string = "{:.3g}".format(param_value)
                param_string += f"{param_name} = {param_value_string}, "
                log_packet[param_name] = param_value
            log_packet["Parameters"] = param_string[:-2]

            # Pick out the states to log and save them
            x_value = x_current[sim_index, plot_x_index].cpu().numpy().item()
            y_value = x_current[sim_index, plot_y_index].cpu().numpy().item()
            log_packet[plot_x_label] = x_value
            log_packet[plot_y_label] = y_value
            log_packet["state"] = x_current[sim_index, :].cpu().detach().numpy()

            x_value = x_current_nn[sim_index, plot_x_index].cpu().numpy().item()
            y_value = x_current_nn[sim_index, plot_y_index].cpu().numpy().item()
            log_packet[plot_x_nn_label] = x_value
            log_packet[plot_y_nn_label] = y_value
            
            results.append(log_packet)
        # Simulate forward using the dynamics
        for i in range(n_sims):
            xdot = system.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_current[i, :].unsqueeze(0),
                params,
            )
            x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()
            print(xdot)
            xdot = nn_dynamics(
                x_current_nn[i, :].unsqueeze(0),
                u_current_nn[i, :].unsqueeze(0),
            ).detach()
            x_current_nn[i, :] = x_current_nn[i, :] + delta_t * xdot.squeeze()
            print(xdot)
    return pd.DataFrame(results)




def plot(results_df: pd.DataFrame):
    num_plots = 1
    fig, ax = plt.subplots(1, num_plots)
    fig.set_size_inches(9 * num_plots, 6)

    rollout_ax = ax
    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        sim_mask = results_df["Simulation"] == sim_index
        rollout_ax.plot(
            results_df[sim_mask][plot_x_nn_label].to_numpy(), 
            results_df[sim_mask][plot_y_nn_label].to_numpy(),
            linestyle="-",
            # marker="+",
            markersize=5,
            color=sns.color_palette(n_colors=100)[plot_idx],
        )
        rollout_ax.set_xlabel(plot_x_label)
        rollout_ax.set_ylabel(plot_y_label)
    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        sim_mask = results_df["Simulation"] == sim_index
        rollout_ax.plot(
            results_df[sim_mask][plot_x_label].to_numpy(), 
            results_df[sim_mask][plot_y_label].to_numpy(),
            linestyle="-",
            # marker="+",
            markersize=5,
            color=sns.color_palette(n_colors=100)[plot_idx+3],
        )
        rollout_ax.set_xlabel(plot_x_label)
        rollout_ax.set_ylabel(plot_y_label)
    plt.savefig("pendulum.png")
   
   
    
x_start = torch.tensor([
    [1.5, 1.5],
    [0.9, 1.5],
    [0.3, 1.5]]
)

nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}

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

    
dir_path = "/home/ny0221/neural_cbf/neural_cbf/training/checkpoints/"
nn_dynamics = DynamicsModel.load_from_checkpoint(dir_path + "pendulum.ckpt", n_dims=simulator.n_dims, n_controls=simulator.n_controls, control_limits=simulator.control_limits)    

log_file = "/home/ny0221/neural_cbf/neural_cbf/training/lightning_logs/version_3/checkpoints/model.ckpt"
neural_controller = NeuralCBFController.load_from_checkpoint(log_file, dynamics_model=nn_dynamics, datamodule=data_module, system=simulator)
   
experiment = CBFVerificationExperiment()
results_df = experiment.run(neural_controller) 
experiment.plot(neural_controller, results_df)
import pdb
pdb.set_trace()
results_df = run(x_start, nominal_params, simulator, neural_controller, nn_dynamics)
plot(results_df)
