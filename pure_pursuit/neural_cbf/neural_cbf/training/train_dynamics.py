import pytorch_lightning as pl

import torch
from scripts.create_data import F110System,parse_args
from scripts.pure_pursuit_control import PurePursuitController
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy

controller_period = 0.01
simulation_dt = 0.001
from matplotlib import pyplot as plt

from pdb import set_trace as bp

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
        cond1 = np.logical_and(steering_angle <= s_min,steering_velocity <= 0)
        cond2 = np.logical_and(steering_angle >= s_max,steering_velocity >= 0)
        cond3 = steering_velocity <= sv_min
        cond4 = steering_velocity >= sv_max

        cond = np.logical_or(cond1, cond2)
        cond3 = np.logical_and(np.logical_not(cond), cond3)
        cond4 = np.logical_and(np.logical_not(cond), cond4)

        steering_velocity[cond] = 0.
        steering_velocity[cond3] = sv_min
        steering_velocity[cond4] = sv_max

        # if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        #     steering_velocity = 0.
        # elif steering_velocity <= sv_min:
        #     steering_velocity = sv_min
        # elif steering_velocity >= sv_max:
        #     steering_velocity = sv_max

        return np.expand_dims(steering_velocity,axis = 1)

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
        accl_mod = deepcopy(accl)

        pos_limit = np.zeros_like(accl)
        pos_limit[vel > v_switch] = a_max*v_switch/vel[vel > v_switch]
        pos_limit[vel <= v_switch] = a_max

        # if vel > v_switch:
        #     pos_limit = a_max*v_switch/vel
        # else:
        #     pos_limit = a_max

        # accl limit reached?
        cond1 = np.logical_and(vel <= v_min, accl <= 0) 
        cond2 = np.logical_and(vel >= v_max, accl >= 0)
        cond = np.logical_or(cond1, cond2)
        accl_mod[cond] = 0.

        cond3 = np.logical_and(np.logical_not(cond), accl <= -a_max)
        cond4 = np.logical_and(np.logical_not(cond), accl >= pos_limit)

        accl_mod[cond3] = -a_max
        accl_mod[cond4] = pos_limit[cond4]
        
        # if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        #     accl = 0.
        # elif accl <= -a_max:
        #     accl = -a_max
        # elif accl >= pos_limit:
        #     accl = pos_limit

        return np.expand_dims(accl_mod,axis = 1)

    def vehicle_dynamics_ks(self, x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
        """
        Single Track Kinematic Vehicle Dynamics.

            Args:
                x (numpy.ndarray (n, 5)): vehicle state vector (x1, x2, x3, x4, x5)
                    x1: x position in global coordinates
                    x2: y position in global coordinates
                    x3: steering angle of front wheels
                    x4: velocity in x direction
                    x5: yaw angle
                u (numpy.ndarray (n, 2)): control input vector (u1, u2)
                    u1: steering angle of front wheels
                    u2: velocity

            Returns:
                f (numpy.ndarray): (n,5) right hand side of differential equations
                g (numpy.ndarray): (n,5,2) control matrix
        """
        # wheelbase
        lwb = lf + lr
        n,state_dim = x.shape
        n,control_dim = u_init.shape

        #getting acelerations and steering velocities
        accl, sv = self.pid(u_init[:,0], u_init[:,1], x[:,3], x[:,2],
         self.vehicle_params['sv_max'], self.vehicle_params['a_max'], 
         self.vehicle_params['v_max'], self.vehicle_params['v_min'])
        # constraints
        # u = np.array([self.steering_constraint(x[:,2],sv, s_min, s_max, sv_min, sv_max), 
        # self.accl_constraints(x[:,3], accl, v_switch, a_max, v_min, v_max)])
        u = np.concatenate((self.steering_constraint(x[:,2],sv, s_min, s_max, sv_min, sv_max),
            self.accl_constraints(x[:,3], accl, v_switch, a_max, v_min, v_max)), axis = 1)
        
        u = torch.tensor(u, dtype = torch.float32)

        # system dynamics
        f = np.zeros((n,state_dim))
        f[:,0] = x[:,3]*np.cos(x[:,4])
        f[:,1] = x[:,3]*np.sin(x[:,4])
        f[:,4] = x[:,3]/lwb*np.tan(x[:,2])

        # f = np.array([x[3]*np.cos(x[4]),
        #     x[3]*np.sin(x[4]),
        #     0,
        #     0,
        #     x[3]/lwb*np.tan(x[2])])

        f = torch.tensor(f,dtype = torch.float32)

        g = np.zeros((n,state_dim,control_dim))
        g[:,2,0] = u[:,0]/u_init[:,0]
        g[:,3,1] = u[:,1]/u_init[:,1]
        
        # g = np.array([[0, 0, u[0]/u_init[0], 0, 0],
        #               [0, 0, 0, u[1]/u_init[1], 0]]).T
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
        # if np.fabs(steer_diff) > 1e-4:
        #     sv = (steer_diff / np.fabs(steer_diff)) * max_sv
        # else:
        #     sv = 0.0

        # vectorized
        sv = np.zeros_like(steer_diff)
        cond1 = np.fabs(steer_diff) > 1e-4
        sv[cond1] = (steer_diff[cond1] / np.fabs(steer_diff[cond1])) * max_sv

        # accl
        vel_diff = speed - current_speed
        # #currently forward
        # if current_speed > 0.:
        #     if (vel_diff > 0):
        #         # accelerate
        #         kp = 10.0 * max_a / max_v
        #         accl = kp * vel_diff
        #     else:
        #         # braking
        #         kp = 10.0 * max_a / (-min_v)
        #         accl = kp * vel_diff
        # # currently backwards
        # else:
        #     if (vel_diff > 0):
        #         # braking
        #         kp = 2.0 * max_a / max_v
        #         accl = kp * vel_diff
        #     else:
        #         # accelerating
        #         kp = 2.0 * max_a / (-min_v)
        #         accl = kp * vel_diff

        # vectorized implementation
        kps = np.zeros_like(vel_diff)

        cond1 = np.logical_and(current_speed > 0. , vel_diff > 0.)
        kp = 10.0 * max_a / max_v
        kps[np.where(cond1)] = kp

        cond2 = np.logical_and(current_speed > 0. , vel_diff <= 0.)
        kp = 10.0 * max_a / (-min_v)
        kps[np.where(cond2)] = kp
    

        cond3 = np.logical_and(current_speed <= 0. , vel_diff > 0.)
        kp = 2.0 * max_a / max_v
        kps[np.where(cond3)] = kp

        cond4 = np.logical_and(current_speed <= 0. , vel_diff <= 0.)
        kp = 2.0 * max_a / (-min_v)
        kps[np.where(cond4)] = kp

        accl = kps * vel_diff
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
    all_controls = []

    while (np.linalg.norm(start_np[:2] - goal[:2]) > 0.1):
        orientation = R.from_euler('z', start_np[4], degrees=False).as_quat()
        try:
            control = system.controller.compute_control(start_np[0],start_np[1],orientation)  
        except:
            break
        f,g = dynamics(np.expand_dims(start, axis=0), np.expand_dims(control,axis = 0))
        all_controls.append(control)
        U = torch.tensor(np.array([[control[0], control[1]]]),dtype = torch.float32)
        U = torch.unsqueeze(U, dim=-1)
        #print(U.shape)

        start_vec = torch.unsqueeze(start, dim=0) + dt*(f + torch.bmm(g, U).squeeze(-1))
        start = start_vec.squeeze()
        #print(start)
        all_states.append(start.detach().numpy())
        start_np = start.detach().numpy()

    # compare original trajectory with the simulated trajectory
    all_states = np.array(all_states)
    all_controls = np.array(all_controls)

    print(all_states.shape)
    plt.plot(actual_waypoints[:,0], actual_waypoints[:,1], label = 'reference')
    #plt.figure()
    plt.plot(all_states[:,0], all_states[:,1], label = 'simulated')
    #plot goal points 
    plt.plot(goal[0], goal[1], 'ro', label = 'goal')
    plt.plot(start_state[0], start_state[1], 'go', label = 'start')
    plt.legend()
    plt.show()

    # Test dynamics in a batch setting
    f,g = dynamics(all_states[:-1,:], all_controls)
    print(f.shape)
    print(g.shape)
    print(all_states.shape)
    print(all_controls.shape)
    state_dash = torch.tensor(all_states[:-1,:],dtype = torch.float32)\
                 + dt * (f + torch.bmm(g,torch.tensor(all_controls,dtype = torch.float32).unsqueeze(-1)).squeeze(-1))

    assert np.linalg.norm(state_dash-all_states[1:,:])<1e-2

    
if __name__ == "__main__":
    args = parse_args()

    main(args)
