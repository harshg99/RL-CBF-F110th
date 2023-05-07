import pytorch_lightning as pl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR
from scripts.create_data import F110System,parse_args
from scripts.pure_pursuit_control import PurePursuitController
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

controller_period = 0.01
simulation_dt = 0.001
from matplotlib import pyplot as plt
class F110DynamicsModel:
    def __init__(self, n_dims, n_controls, vehicle_parameters=None,params = None):
        self.n_dims = n_dims
        self.n_controls = n_controls       
        if vehicle_parameters is None:
            self.vehicle_params = {'mu': 1.0489, 
            'C_Sf': 4.718,
            'C_Sr': 5.4562,
            'lf': 0.15875,
            'lr': 0.17145,
            'h': 0.074,
            'm': 3.74,
            'I': 0.04712, 
            's_min': -0.4189,
            's_max': 0.4189,
            'sv_min': -3.2,
            'sv_max': 3.2,
            'v_switch': 7.319,
            'a_max': 6.51,
            'v_min':-5.0,
            'v_max': 8.0,
            'width': 0.31,
            'length': 0.58}

        if params is None:
            params = {'dt': 1e-3}
    
    def __call__(self, x, u):
        '''
        X,y,theta_steer,v,yaw -> f(X,y,theta_steer,v,yaw) + g(X,y,theta_steer,v,yaw)*u
        '''

        state = x

        f,g = self.vehicle_dynamics_ks(
            state,
            np.array(u),
            self.vehicle_params['mu'],
            self.vehicle_params['C_Sf'],
            self.vehicle_params['C_Sr'],
            self.vehicle_params['lf'],
            self.vehicle_params['lr'],
            self.vehicle_params['h'],
            self.vehicle_params['m'],
            self.vehicle_params['I'],
            self.vehicle_params['s_min'],
            self.vehicle_params['s_max'],
            self.vehicle_params['sv_min'],
            self.vehicle_params['sv_max'],
            self.vehicle_params['v_switch'],
            self.vehicle_params['a_max'],
            self.vehicle_params['v_min'],
            self.vehicle_params['v_max'])
        
        
        return f,g

    def steering_constraint(self, steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
        """
        Steering constraints, adjusts the steering velocity based on constraints

            Args:
                steering_angle (float): current steering_angle of the vehicle
                steering_velocity (float): unconstraint desired steering_velocity
                s_min (float): minimum steering angle
                s_max (float): maximum steering angle
                sv_min (float): minimum steering velocity
                sv_max (float): maximum steering velocity

            Returns:
                steering_velocity (float): adjusted steering velocity
        """

        # constraint steering velocity
        if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
            steering_velocity = 0.
        elif steering_velocity <= sv_min:
            steering_velocity = sv_min
        elif steering_velocity >= sv_max:
            steering_velocity = sv_max

        return steering_velocity

    def accl_constraints(self, vel, accl, v_switch, a_max, v_min, v_max):
        """
        Acceleration constraints, adjusts the acceleration based on constraints

            Args:
                vel (float): current velocity of the vehicle
                accl (float): unconstraint desired acceleration
                v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
                a_max (float): maximum allowed acceleration
                v_min (float): minimum allowed velocity
                v_max (float): maximum allowed velocity

            Returns:
                accl (float): adjusted acceleration
        """

        # positive accl limit
        if vel > v_switch:
            pos_limit = a_max*v_switch/vel
        else:
            pos_limit = a_max

        # accl limit reached?
        if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
            accl = 0.
        elif accl <= -a_max:
            accl = -a_max
        elif accl >= pos_limit:
            accl = pos_limit

        return accl

    def vehicle_dynamics_ks(self, x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
        """
        Single Track Kinematic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                u (numpy.ndarray (2, )): control input vector (u1, u2)
                    u1: steering angle of front wheels
                    u2: velocity

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # wheelbase
        lwb = lf + lr
        
        #getting acelerations and steering velocities
        accl, sv = self.pid(u_init[0], u_init[1], x[3], x[2],
         self.vehicle_params['sv_max'], self.vehicle_params['a_max'], 
         self.vehicle_params['v_max'], self.vehicle_params['v_min'])

        # constraints
        u = np.array([self.steering_constraint(x[2],sv, s_min, s_max, sv_min, sv_max), 
        self.accl_constraints(x[3], accl, v_switch, a_max, v_min, v_max)])
        
        u = torch.tensor(u, dtype = torch.float32)

        # system dynamics
        f = np.array([x[3]*np.cos(x[4]),
            x[3]*np.sin(x[4]),
            0,
            0,
            x[3]/lwb*np.tan(x[2])])
        f = torch.tensor(f,dtype = torch.float32)

        g = np.array([[0, 0, u[0]/u_init[0], 0, 0],
                      [0, 0, 0, u[1]/u_init[1], 0]]).T
        g = torch.tensor(g,dtype = torch.float32)
        return (f,g)

    def pid(self, speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
        """
        Basic controller for speed/steer -> accl./steer vel.

            Args:
                speed (float): desired input speed
                steer (float): desired input steering angle

            Returns:
                accl (float): desired input acceleration
                sv (float): desired input steering velocity
        """
        # steering
        steer_diff = steer - current_steer
        if np.fabs(steer_diff) > 1e-4:
            sv = (steer_diff / np.fabs(steer_diff)) * max_sv
        else:
            sv = 0.0

        # accl
        vel_diff = speed - current_speed
        # currently forward
        if current_speed > 0.:
            if (vel_diff > 0):
                # accelerate
                kp = 10.0 * max_a / max_v
                accl = kp * vel_diff
            else:
                # braking
                kp = 10.0 * max_a / (-min_v)
                accl = kp * vel_diff
        # currently backwards
        else:
            if (vel_diff > 0):
                # braking
                kp = 2.0 * max_a / max_v
                accl = kp * vel_diff
            else:
                # accelerating
                kp = 2.0 * max_a / (-min_v)
                accl = kp * vel_diff

        return accl, sv

def main(args):
    # Test dynamics

    # Load the reference trajectory here and compute controls from the purepuruit controller
    # Simulate dynamics with the purepursuit controller
    # compare original trajectory with the simulated trajectory

    # Load the reference trajectory here and compute controls from the purepuruit controller
    system = F110System(args)
    dt = 0.01
    actual_waypoints = system.controller.waypoints.copy()
    actual_waypoints = np.array(actual_waypoints)
    goal = system.controller.goal_point.copy()

    start_state = np.array([actual_waypoints[0,0], actual_waypoints[0,1], 0, 0, 0])
    start_state = torch.tensor(start_state, dtype = torch.float32)
    dynamics = F110DynamicsModel(n_dims = 5, n_controls = 2)
    # Simulate dynamics with the purepursuit controller
    start = deepcopy(start_state)
    all_states = [deepcopy(start.detach().numpy())]
    start_np = start.detach().numpy()
    while (np.linalg.norm(start_np[:2] - goal[:2]) > 0.1):
        orientation = R.from_euler('z', start_np[4], degrees=False).as_quat()
        try:
            control = system.controller.compute_control(start_np[0],start_np[1],orientation)
        except:
            break
        f,g = dynamics(start, control)
        # print(f.shape)
        # print(g.shape)
        start = start + dt*(f + g@np.array([control[0], control[1]]))
        all_states.append(start.detach().numpy())
        start_np = start.detach().numpy()

    # compare original trajectory with the simulated trajectory
    all_states = np.array(all_states)
    print(all_states.shape)
    plt.plot(actual_waypoints[:,0], actual_waypoints[:,1], label = 'reference')
    #plt.figure()
    plt.plot(all_states[:,0], all_states[:,1], label = 'simulated')
    #plot goal points 
    plt.plot(goal[0], goal[1], 'ro', label = 'goal')
    plt.plot(start_state[0], start_state[1], 'go', label = 'start')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    args = parse_args()

    main(args)
