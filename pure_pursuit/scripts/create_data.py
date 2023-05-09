import numpy as np
import csv
from scripts.pure_pursuit_control import PurePursuitController
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import argparse
# TODO CHECK: include needed ROS msg type headers and lib
TEST_DATA = True
from matplotlib import pyplot as plt

class F110System:
    def __init__(self, args_main):
        self.args = args_main
        args = dict()
        args['lookahead_distance'] = 1.25
        args['velocity'] = 2.5
        args['speed_lookahead_distance'] = 1.50
        args['brake_gain'] = 0.4
        args['wheel_base'] = 0.33
        args['visualize'] = False
        args['curvature_thresh'] = 0.1
        args['acceleration_lookahead_distance'] = 5.0
        args['accel_gain'] = 0.0
        args = argparse.Namespace(**args)

        filename = "scripts/raceline_centre_half.csv"

        positions = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            print(csv_reader)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                else:
                    positions.append([float(row[0]), float(row[1])])
                line_count += 1
        
        waypoints = np.array(positions)
        print(f'Processed {line_count} lines.')
        self.controller = PurePursuitController(args, waypoints.copy())

    def within_bounds(self,x,y):

            ## below are the coefficients a, b, and c for all of the outer track lines L and centerlines CL for standard form ax + by = c

            al1 = -2.2184
            bl1 = 1
            cl1 = 38.0722

            acl1 = -2.2824
            bcl1 = 1
            ccl1 = 36.6968

            al2 = 0.4424
            bl2 = 1
            cl2 = 10.6387

            acl2 = 0.4424
            bcl2 = 1
            ccl2 = 9.8021

            al3 = -2.4726
            bl3 = 1
            cl3 = -24.8661

            acl3 = -2.3683
            bcl3 = 1
            ccl3 = -21.6509

            al4 = 0.4513
            bl4 = 1
            cl4 = -0.4159

            acl4 = 0.4484
            bcl4 = 1
            ccl4 = 0.4599

            ## below is the max acceptable distance from the centerlines
            maxDistance = self.args.margin

            ## first, check if the point is within the outer track
            withinOutterTrack = (al1*x + bl1*y <= cl1) and (al2*x + bl2*y <= cl2) and (al3*x + bl3*y >= cl3) and (al4*x + bl4*y >= cl4)


            ## now calculate the distances from the point to each of the four centerlines
            d1 = abs(acl3*x + bcl3*y - ccl3) / ((acl3**2 + bcl3**2)**0.5)
            d2 = abs(acl4*x + bcl4*y - ccl4) / ((acl4**2 + bcl4**2)**0.5)
            # d3 = abs(al3*x + bl3*y - cl3) / ((al3**2 + bl3**2)**0.5)
            # d4 = abs(al4*x + bl4*y - cl4) / ((al4**2 + bl4**2)**0.5)


            ## now check if the point is within +- 0.25m of any of the centerlines
            withinCenterlines = (d1 <= maxDistance) or (d2 <= maxDistance) #or (d3 <= maxDistance) or (d4 <= maxDistance)

            return withinOutterTrack and withinCenterlines
        
    def ttc_bounds(self, x,y, yaw, vel):
        ## below are the coefficients a, b, and c for all of the outer track lines L and centerlines CL for standard form ax + by = c

        al3 = -2.4726
        bl3 = 1
        cl3 = -24.8661

        ail3 = -2.4726
        bil3 = 1
        cil3 = -19.0661

        al4 = 0.4513
        bl4 = 1
        cl4 = -0.4159

        ail4 = 0.4513
        bil4 = 1
        cil4 = 1.32

        ## below is the max acceptable distance from the centerlines
        maxTTC = self.args.max_ttc

        # compute the distance to each of the centerlines

        
        ## now calculate the distances from the point to each of the four centerlines
        d1 = -(al3*x + bl3*y - cl3) / (al3*np.cos(yaw) + bl3*np.sin(yaw))
        d2 = -(al4*x + bl4*y - cl4) / (al4*np.cos(yaw) + bl4*np.sin(yaw))
        d3 = -(ail3*x + bil3*y - cil3) / (ail3*np.cos(yaw) + bil3*np.sin(yaw))
        d4 = -(ail4*x + bil4*y - cil4) / (ail4*np.cos(yaw) + bil4*np.sin(yaw))

        if d1 < 0:
            d1 = 100
        if d2 < 0:
            d2 = 100
        if d3 < 0:
            d3 = 100
        if d4 < 0:
            d4 = 100

        ttc1 = d1/vel
        ttc2 = d2/vel
        ttc3 = d3/vel
        ttc4 = d4/vel

        # check if any of the ttcs are less than the maxTTC
        withinTTC = (ttc1 <= maxTTC) or (ttc2 <= maxTTC) or (ttc3 <= maxTTC) or (ttc4 <= maxTTC)
        return np.logical_not(withinTTC)



    def generate_data(self, num_samples = None):

        goal = self.controller.goal_point
        start_point = self.controller.start_point

        if num_samples is None:
            num_samples = self.args.num_samples

        al3 = -2.4726
        bl3 = 1
        cl3 = -24.8661

        ail3 = -2.4726
        bil3 = 1
        cil3 = -19.0661

        al4 = 0.4513
        bl4 = 1
        cl4 = -0.4159

        ail4 = 0.4513
        bil4 = 1
        cil4 = 1.32

        al1 = -2.2184
        bl1 = 1
        cl1 = 38.0722

        al2 = 0.4424
        bl2 = 1
        cl2 = 10.6387


        safe_data = {'states': [], 'controls': []}
        unsafe_data = {'states':[], 'controls': []}

        for i in tqdm(range(num_samples)):
            valid_sample = False
            while(not valid_sample):
                valid_position = False
                while(not valid_position):
                    x = np.random.uniform(args.bounds_lower, args.bounds_upper)
                    y = np.random.uniform(args.bounds_lower, args.bounds_upper)
                    withinOutterTrack = (al1*x + bl1*y <= cl1) and (al2*x + bl2*y <= cl2) and (al3*x + bl3*y >= cl3) and (al4*x + bl4*y >= cl4)
                    withininnertrack = (ail3*x + bil3*y <= cil3) or (ail4*x + bil4*y <= cil4)
                    valid_position = withinOutterTrack and withininnertrack

                vel = np.random.uniform(args.vel_lower, args.vel_upper)
                yaw = np.random.uniform(-2*np.pi/3, 2*np.pi/3)
                steer = np.random.uniform(-args.steering_max,args.steering_max)

                orientation = R.from_euler('z', yaw).as_quat()

                try:
                    v_con, theta_con = self.controller.compute_control(x,y,orientation)
                    if v_con < 0:
                        v_con = 1e-3
                    elif v_con > self.args.vel_upper:
                        v_con = self.args.vel_upper
                    safe = self.within_bounds(x, y) and self.ttc_bounds(x, y, yaw, vel)

                    if safe:
                        safe_data['states'].append([x, y, steer, vel, yaw])
                        safe_data['controls'].append([v_con, theta_con])
                    else:
                        unsafe_data['states'].append([x, y, steer, vel, yaw])
                        unsafe_data['controls'].append([v_con, theta_con])
                    valid_sample = True
                except:
                    valid_sample = False
                    #print("Invalid sample at{}, trying again".format(i))

        safe_data['states'] = np.array(safe_data['states'])
        safe_data['controls'] = np.array(safe_data['controls'])
        unsafe_data['states'] = np.array(unsafe_data['states'])
        unsafe_data['controls'] = np.array(unsafe_data['controls'])
        safe_data = np.array(safe_data)
        unsafe_data = np.array(unsafe_data)
        metadata = np.array({'goal': goal, 'start_point': start_point})
        return safe_data, unsafe_data, metadata

    def save_data(self, safe_data, unsafe_data, metadata):
        if not os.path.exists(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        np.save(self.args.save_dir + '/safe_trajectory.npy', safe_data, allow_pickle=True)
        np.save(self.args.save_dir + '/unsafe_trajectory.npy', unsafe_data, allow_pickle= True)
        np.save(self.args.save_dir + '/metadata.npy', metadata, allow_pickle=True)
    
    def load_data(self):
        safe_data = np.load(self.args.save_dir + '/safe_trajectory.npy', allow_pickle=True).item()
        unsafe_data = np.load(self.args.save_dir + '/unsafe_trajectory.npy', allow_pickle=True).item()
        metadata = np.load(self.args.save_dir + '/metadata.npy', allow_pickle=True).item()
        return safe_data, unsafe_data, metadata
    
    def visualize_data(self):
        # XY visualization with safe and unsafe data
        safe_data, unsafe_data, metadata = self.load_data()
        safe_states = safe_data['states']
        safe_controls = safe_data['controls']
        unsafe_states = unsafe_data['states']
        unsafe_controls = unsafe_data['controls']
        goal = metadata['goal']
        start_point = metadata['start_point']

        plt.figure()
        plt.scatter(safe_states[:,0], safe_states[:,1], c='g', label='Safe')
        plt.scatter(unsafe_states[:,0], unsafe_states[:,1], c='r', label='Unsafe',alpha=0.5)
        plt.scatter(goal[0], goal[1], c='b', label='Goal')
        plt.scatter(self.controller.waypoints[:,0], self.controller.waypoints[:,1], c='k')
        plt.scatter(start_point[0], start_point[1], c='k', label='Start')
        plt.legend()
        #plt.show()
        plt.savefig(self.args.save_dir + '/xy_visualization.png')

        # Safe XY visualization with velocity and heading
        plt.figure()
        plt.quiver(safe_states[:,0], safe_states[:,1], np.cos(safe_states[:,4]), np.sin(safe_states[:,4]), safe_states[:,3], cmap='jet')
        plt.colorbar()
        #plt.show()
        plt.savefig(self.args.save_dir + '/safe_xy_visualization.png')

        # Unsafe XY visualization with velocity and heading
        plt.figure()
        plt.quiver(unsafe_states[:,0], unsafe_states[:,1], np.cos(unsafe_states[:,4]), np.sin(unsafe_states[:,4]), unsafe_states[:,3], cmap='jet')
        plt.colorbar()
        #plt.show()
        plt.savefig(self.args.save_dir + '/unsafe_xy_visualization.png')

        # plt waypoints
        plt.figure()
        plt.scatter(self.controller.waypoints[:,0], self.controller.waypoints[:,1], c='k')
        #splt.show()
        plt.savefig(self.args.save_dir + '/waypoints.png')

        # Histogram of steering angles
        plt.figure()
        plt.hist(np.concatenate((safe_controls[:,-1], unsafe_controls[:,-1]), axis = 0), bins=100)
        plt.title('Steering angles')
        #plt.show()
        plt.savefig(self.args.save_dir + '/steering_angles.png')

        # Histogram of velocities
        plt.figure()
        plt.hist(np.concatenate((safe_controls[:,0],unsafe_controls[:,0]), axis=0), bins=100)
        plt.title('Velocities')
        #plt.show()
        plt.savefig(self.args.save_dir + '/velocities.png')







def parse_args(return_args=True):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1000000)
    parser.add_argument('--bounds_lower', type=float, default=-14)
    parser.add_argument('--bounds_upper', type=float, default=14)
    parser.add_argument('--steering_max', type=float, default=0.50)
    parser.add_argument('--margin', type=float, default=0.50)
    parser.add_argument('--max_ttc', type=float, default=0.4)
    parser.add_argument('--vel_lower', type=float, default=0.00)
    parser.add_argument('--vel_upper', type=float, default=2.6)
    parser.add_argument('--filename', type=str, default='waypoints.csv') 
    parser.add_argument('--save_dir', type=str, default='trajectory_data/')
    if return_args:
        args = parser.parse_args()
        return args
    else:
        return parser

if __name__ == '__main__':
    args = parse_args()
    system = F110System(args)
    safe_data,unsafe_data,metadata =  system.generate_data(args.num_samples)
    system.save_data(safe_data, unsafe_data, metadata)
    if TEST_DATA:
        system.visualize_data()

