from neural_cbf.systems import KSCar

import pytorch_lightning as pl

import torch
from torch.utils.data import TensorDataset, random_split, DataLoader
from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR

controller_period = 0.01
simulation_dt = 0.001

class F110DynamicsModel:
    def __init__(self, n_dims, n_controls, control_limits, vehicle_parameters,params = None):
        super(F110DynamicsModel, self).__init__()
        self.layer1 = torch.nn.Linear(n_dims + n_controls, 32)
        self.layer2 = torch.nn.Linear(32, 64)
        self.layer3 = torch.nn.Linear(64, n_dims)
        self.loss_fn = torch.nn.MSELoss()
        self.n_dims = n_dims
        self.n_controls = n_controls       
        self.control_limits = control_limits 

        if vehicle_parameters is None:
            self.vehicle_parameters = {'mu': 1.0489, 
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
            np.array([sv, accl]),
            self.params['mu'],
            self.params['C_Sf'],
            self.params['C_Sr'],
            self.params['lf'],
            self.params['lr'],
            self.params['h'],
            self.params['m'],
            self.params['I'],
            self.params['s_min'],
            self.params['s_max'],
            self.params['sv_min'],
            self.params['sv_max'],
            self.params['v_switch'],
            self.params['a_max'],
            self.params['v_min'],
            self.params['v_max'])
        
        
        return f,g

    def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
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

    def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
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

    def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
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
                    u1: steering angle velocity of front wheels
                    u2: longitudinal acceleration

            Returns:
                f (numpy.ndarray): right hand side of differential equations
        """
        # wheelbase
        lwb = lf + lr
        # constraints
        u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])
        u = torch.tensor(u, dtype = torch.float32)

        # system dynamics
        f = np.array([x[3]*np.cos(x[4]),
            x[3]*np.sin(x[4]),
            0,
            0,
            x[3]/lwb*np.tan(x[2])])
        f = torch.tensor(f,dtype = torch.float32)

        g = np.array([0, 0, u[0]/u_init[0], 0, 0,
                      0, 0, 0, u[1]/u_init[1], 0]).T
        return (f,g)

    def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
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
