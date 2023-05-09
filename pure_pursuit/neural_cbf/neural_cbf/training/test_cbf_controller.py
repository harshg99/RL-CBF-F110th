import matplotlib.pyplot as plt
import torch
from neural_cbf.neural_cbf.datamodules import F110DataModule

from argparse import ArgumentParser
from scripts.create_data import F110System,parse_args

from neural_cbf.neural_cbf.training import NeuralCBFController, F110DynamicsModel,ILController

import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

from cbf_grid_verification import CBFVerificationExperiment

plot_x_label = '$\\theta$'
plot_y_label = '$\\dot{\\theta}$'

plot_x_nn_label = '$\\theta$_nn'
plot_y_nn_label = '$\\dot{\\theta}$_nn'

plot_x_index = 0
plot_y_index = 1


# def run(x_start, params, system, controller, nn_dynamics):
#     num_timesteps = 1000
#     delta_t = 0.01
#     x_current = x_start
#     x_current_nn = x_start.clone()
#     n_sims = x_start.shape[0]
#     results = []
#     for tstep in range(num_timesteps):
#         # Get the control input at the current state if it's time
#         if tstep % 5 == 0:
#             #u_current = controller.u(x_current)
#             u_current = controller.policy_net(x_current).detach()
#             u_current_nn = controller.policy_net(x_current_nn).detach()
#         # Log the current state and control for each simulation
#         for sim_index in range(n_sims):
#             log_packet = {"t": tstep * delta_t, "Simulation": str(sim_index)}
#
#             # Include the parameters
#             param_string = ""
#             for param_name, param_value in params.items():
#                 param_value_string = "{:.3g}".format(param_value)
#                 param_string += f"{param_name} = {param_value_string}, "
#                 log_packet[param_name] = param_value
#             log_packet["Parameters"] = param_string[:-2]
#
#             # Pick out the states to log and save them
#             x_value = x_current[sim_index, plot_x_index].cpu().numpy().item()
#             y_value = x_current[sim_index, plot_y_index].cpu().numpy().item()
#             log_packet[plot_x_label] = x_value
#             log_packet[plot_y_label] = y_value
#             log_packet["state"] = x_current[sim_index, :].cpu().detach().numpy()
#
#             x_value = x_current_nn[sim_index, plot_x_index].cpu().numpy().item()
#             y_value = x_current_nn[sim_index, plot_y_index].cpu().numpy().item()
#             log_packet[plot_x_nn_label] = x_value
#             log_packet[plot_y_nn_label] = y_value
#
#             results.append(log_packet)
#         # Simulate forward using the dynamics
#         for i in range(n_sims):
#             xdot = system.closed_loop_dynamics(
#                 x_current[i, :].unsqueeze(0),
#                 u_current[i, :].unsqueeze(0),
#                 params,
#             )
#             x_current[i, :] = x_current[i, :] + delta_t * xdot.squeeze()
#             print(xdot)
#             xdot = nn_dynamics(
#                 x_current_nn[i, :].unsqueeze(0),
#                 u_current_nn[i, :].unsqueeze(0),
#             ).detach()
#             x_current_nn[i, :] = x_current_nn[i, :] + delta_t * xdot.squeeze()
#             print(xdot)
#     return pd.DataFrame(results)



def run(start_state, args, system, neural_controller, il_controller, dynamics, data,goal):
    # Simulate dynamics with both the purepursuit controller and the neuralnet controller

    # Simulating purepuruit controller
    start = deepcopy(start_state)
    all_states_expert = [deepcopy(start.detach().numpy())]
    start_np = start.detach().numpy()
    all_controls_expert = []
    dt = 0.01
    max_steps = 1200

    step = 0
    while (np.linalg.norm(start_np[:2] - goal[:2]) > 0.1) and step < max_steps:
        orientation = R.from_euler('z', start_np[4], degrees=False).as_quat()
        try:
            control = system.controller.compute_control(start_np[0],start_np[1],orientation)
        except:
            break
        f,g = dynamics(np.expand_dims(start, axis=0), np.expand_dims(control,axis = 0))
        all_controls_expert.append(control)
        U = torch.tensor(np.array([[control[0], control[1]]]),dtype = torch.float32)
        U = torch.unsqueeze(U, dim=-1)
        #print(U.shape)

        start_vec = torch.unsqueeze(start, dim=0) + dt*(f + torch.bmm(g, U).squeeze(-1))
        start = start_vec.squeeze()
        #print(start)
        all_states_expert.append(start.detach().numpy())
        start_np = start.detach().numpy()
        step += 1

    # compare original trajectory with the simulated trajectory
    all_states_expert = np.array(all_states_expert)
    all_controls_expert = np.array(all_controls_expert)

    # Simulating neuralnet controller
    start = deepcopy(start_state)
    all_states_nn = [deepcopy(start.detach().numpy())]
    start_np = start.detach().numpy()
    all_controls_nn = []
    dt = 0.01
    step = 0
    while (np.linalg.norm(start_np[:2] - goal[:2]) > 0.1) and step < max_steps:
        try:
            control = neural_controller.policy_net(start)
            control = control.cpu().detach().numpy()
        except:
            break
        all_controls_nn.append(control)
        control = np.expand_dims(control, axis=0)
        f, g = dynamics(np.expand_dims(start_np, axis=0), control)


        # print(U.shape)
        U = torch.tensor(control, dtype=torch.float32).unsqueeze(-1)
        start_vec = torch.unsqueeze(start, dim=0) + dt * (f + torch.bmm(g, U).squeeze(-1))
        start = start_vec.squeeze()
        # print(start)
        all_states_nn.append(start.detach().numpy())
        start_np = start.detach().numpy()
        step += 1

    # compare original trajectory with the simulated trajectory
    all_states_nn = np.array(all_states_nn)
    all_controls_nn = np.array(all_controls_nn)

    start = deepcopy(start_state)
    all_states_il = [deepcopy(start.detach().numpy())]
    start_np = start.detach().numpy()
    all_controls_il = []
    dt = 0.01
    step = 0
    while (np.linalg.norm(start_np[:2] - goal[:2]) > 0.1) and step < max_steps:
        try:
            control = il_controller.policy_net(start)
            control = control.cpu().detach().numpy()
        except:
            break
        all_controls_il.append(control)
        control = np.expand_dims(control, axis=0)
        f, g = dynamics(np.expand_dims(start_np, axis=0), control)


        # print(U.shape)
        U = torch.tensor(control, dtype=torch.float32).unsqueeze(-1)
        start_vec = torch.unsqueeze(start, dim=0) + dt * (f + torch.bmm(g, U).squeeze(-1))
        start = start_vec.squeeze()
        # print(start)
        all_states_il.append(start.detach().numpy())
        start_np = start.detach().numpy()
        step += 1

    actual_waypoints = system.controller.waypoints.copy()
    actual_waypoints = np.array(actual_waypoints)

    # compare original trajectory with the simulated trajectory
    all_states_il = np.array(all_states_il)
    all_controls_il = np.array(all_controls_il)

    print(all_states_expert.shape)
    plt.plot(actual_waypoints[:,0], actual_waypoints[:,1], label = 'reference')
    #plt.figure()
    plt.plot(all_states_expert[:,0], all_states_expert[:,1], label = 'PurePursuit')
    #plot goal points
    plt.plot(all_states_nn[:,0], all_states_nn[:,1], label = 'NeuralNet')

    plt.plot(all_states_il[:,0], all_states_il[:,1], label = 'Imitation Learning')

    plt.plot(goal[0], goal[1], 'ro', label = 'goal')
    plt.plot(start_state[0], start_state[1], 'go', label = 'start')
    plt.legend()
    plt.show()


def init(args):


    system = F110System(args)
    dt = 0.01
    actual_waypoints = system.controller.waypoints.copy()
    actual_waypoints = np.array(actual_waypoints)
    goal = system.controller.goal_point.copy()

    start_state = np.array([actual_waypoints[0, 0], actual_waypoints[0, 1], 0, 0, 0])
    start_state = torch.tensor(start_state, dtype=torch.float32)
    dynamics = F110DynamicsModel(n_dims=5, n_controls=2)

    datasource = F110DataModule(args,
        model=F110System,
        val_split=0.05,
        batch_size=50,
        quotas={"safe": 0.5, "unsafe": 0.4, "goal": 0.1},
    )

    dir_path = "neural_cbf/training/checkpoints/" + args.version + "/model.ckpt"
    neural_controller = NeuralCBFController.load_from_checkpoint(dir_path, dynamics_model=dynamics,
                                                                 datamodule=datasource, system=system)

    dir_path_il = "neural_cbf/training/checkpoints/" + args.version_il + "/model.ckpt"
    il_controller = ILController.load_from_checkpoint(dir_path_il, dynamics_model=dynamics,
                                                                 datamodule=datasource, system=system)

    return dynamics, system, start_state, goal, neural_controller,il_controller, datasource

if __name__ == "__main__":
    parser  = parse_args(False)
    parser.add_argument('--goal_radius', type=float, default=0.4)
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--version_il', type=str, default='v1')

    args = parser.parse_args()
    dynamics, system, start_state, goal, neural_controller,il_controller, data_module = init(args)
    run(start_state, args, system, neural_controller, il_controller, dynamics, data_module,goal)






# experiment = CBFVerificationExperiment()
# results_df = experiment.run(neural_controller)
# experiment.plot(neural_controller, results_df)

# Simulating the learnt controller with the dynamics model

# pdb.set_trace()
# results_df = run(x_start, nominal_params, simulator, neural_controller, nn_dynamics)
# plot(results_df)
