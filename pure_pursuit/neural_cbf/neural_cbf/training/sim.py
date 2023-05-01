import matplotlib.pyplot as plt
from neural_cbf.systems import InvertedPendulum
from matplotlib.pyplot import figure
import pandas as pd
import torch
import seaborn as sns
plot_x_label = '$\\theta$'
plot_y_label = '$\\dot{\\theta}$'
plot_x_index = 0
plot_y_index = 1
def run(x_start, params, dynamics):
    num_timesteps = 500 
    delta_t = 0.01 
    x_current = x_start
        
    n_sims = x_start.shape[0]
    results = []
    for tstep in range(num_timesteps):
        # Get the control input at the current state if it's time
        #if tstep % controller_update_freq == 0:
            #u_current = controller_under_test.u(x_current)

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

            results.append(log_packet)
        # Simulate forward using the dynamics
        u_current = torch.randn(3, 1)
        for i in range(n_sims):
            xdot = dynamics.closed_loop_dynamics(
                x_current[i, :].unsqueeze(0),
                u_current[i, :].unsqueeze(0),
                params,
            )
            x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()

    return pd.DataFrame(results)




def plot(results_df: pd.DataFrame):
    num_plots = 1
    fig, ax = plt.subplots(1, num_plots)
    fig.set_size_inches(9 * num_plots, 6)

    rollout_ax = ax
    for plot_idx, sim_index in enumerate(results_df["Simulation"].unique()):
        print(plot_idx)
        sim_mask = results_df["Simulation"] == sim_index
        rollout_ax.plot(
            results_df[sim_mask][plot_x_label].to_numpy(),
            results_df[sim_mask][plot_y_label].to_numpy(),
            linestyle="-",
            # marker="+",
            markersize=5,
            color=sns.color_palette(n_colors=100)[plot_idx],
        )
        rollout_ax.set_xlabel(plot_x_label)
        rollout_ax.set_ylabel(plot_y_label)
    
    plt.savefig("pendulum.png")
   
   
    
start_x = torch.tensor([
    [1.5, 1.5],
    [0.9, 1.5],
    [0.3, 1.5]]
)

nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}

simulator = InvertedPendulum(
    nominal_params,
)

results_df = run(start_x, nominal_params, simulator)
plot(results_df)
