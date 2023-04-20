from pdb import set_trace
from scipy.optimize import minimize
import numpy as np

class Optimize:

    def __init__(self, file_name, order):
        self.file_name = file_name
        self.order = order
        # self.max_vel = max_vel
        # self.max_acc = max_acc

    def objective(self,const, expected_arrivals,order = 5):
        W = order + 1
        obj = 0
        for i, Ti in enumerate(expected_arrivals):
            H = np.array([[720 * Ti**5, 360 * Ti**4, 120*Ti**3],
                      [360 * Ti**4, 192 * Ti**3, 72 * Ti**2],
                      [120*Ti**3, 72 * Ti**2, 36 * Ti]])
            x = const[i * W:i * W + 3]
            obj += x.T @ H @ x
        return obj

    def load_tajectory(self, file_name):
        goals = []
        expected_arrivals = []
        with open(file_name, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                print(f'\t{row["x"]} {row["y"]}.')
                goals.append([float(row["x"]), float(row["y"])])
                expected_arrivals.append(float(row["t"]))
                line_count += 1
            print(f'Processed {line_count} lines.')
        return np.array(goals), np.array(expected_arrivals)    

    def get_acc_matrix(self,time,order):
        arr = np.array([(order - j-1)*(order - j)*time**(order - j - 2) for j in range(order + 1)])
        #arr [-2:] = 0
        return arr

    def get_vel_matrix(self,time,order):
        arr = np.array([(order - j)*time**(order - j - 1) for j in range(order + 1)])
        #arr[-1] = 0
        return arr

    def get_jerk_matrix(self,time,order):
        arr = np.array([((order-j-1) *(order-j)*(order-j-2))*time**(order - j - 3) for j in range(order + 1)])
        #arr[-3:] = 0
        return arr

    def constraints_row(self,M,idx,num_segments):
        if not M:
            return None

        H, W = M[0].shape
        constraints = np.zeros((H, W * num_segments))
        for m, id in zip(M, idx):
            constraints[:, id * W:(id + 1) * W] = m
        return constraints

    def optimize_trajectory(self, goals, times):

        segs = lean(goals)
        order = self.order


        assert len(times) == segs

        M_mat = []
        b_mat = []

        # define inequality constrainsts
        M_ineq_mat = []
        b_ineq_mat = []

        # initialize list of 
        constraints_mat = []

        #T()
        #coefficients
        coeffs = np.random.rand((segs) * (order + 1))
        # 1. Initial boundary condition

        loc_cons_t = np.zeros((order+1,))
        loc_cons_t[-1] = 1 
        #(order * [0] + [1])
        loc_cons_tp = np.array([times[0]**(order - i) for i in range(order + 1)])
        vel_cons_t = np.zeros((order+1,))
        vel_cons_t[-2] = 1
        acc_cons_t = np.zeros((order+1,))
        acc_cons_t[-3] = 2

        jerk_cons_t = np.zeros((order+1,))
        jerk_cons_t[-4] = 3*2
        
        M1 = np.vstack((loc_cons_t, loc_cons_tp, jerk_cons_t))
        
        M_mat.append(self.constraints_row([M1], [0], segs))
        b_mat.append(np.array([points[0], points[1], 0]))

        # intermediate points
        for i in range(1, segs-1):
            # boundary conditions
            
            # location constraint
            loc_cons_tp = np.array([times[i]**(order - j) for j in range(order + 1)])
            loc_cons_mat = np.vstack((loc_cons_t, loc_cons_tp))

            M_mat.append(self.constraints_row([loc_cons_mat], [i], segs))
            b_mat.append(np.array([points[i], points[i + 1]]))

            # velocity constraitns at previous time step
            vel_cons_tp = self.get_vel_matrix(times[i-1],order)


            #acceleration constraints at previous timestep
            acc_cons_tp = self.get_acc_matrix(times[i-1],order)

            #jerk constraints at previous timestep
            jerk_cons_tp = self.get_jerk_matrix(times[i-1],order)

        
            vel_cons_mat = np.expand_dims(vel_cons_t,0)
            vel_final_mat = np.expand_dims(vel_cons_tp,0)

            M_mat.append(self.constraints_row([vel_cons_mat, -vel_final_mat], [i, i - 1], segs))
            b_mat.append(np.zeros(1))

            acc_cons_mat = np.expand_dims(acc_cons_t,0)
            acc_final_mat = np.expand_dims(acc_cons_tp,0)

            M_mat.append(self.constraints_row([acc_cons_mat, -acc_final_mat], [i, i - 1], segs))
            b_mat.append(np.zeros(1))



            #Higher order derivatives
            jerk_cons_mat = np.expand_dims(jerk_cons_t,0)
            jerk_final_mat = np.expand_dims(jerk_cons_tp,0)

            M_mat.append(self.constraints_row([jerk_cons_mat, -jerk_final_mat], [i, i - 1], segs))
            b_mat.append(np.zeros(1))

        # 3. Final point of the entire path
        loc_cons_tp = np.array([times[-1]**(order - i) for i in range(order + 1)])

        vel_cons_tp = self.get_vel_matrix(times[-1],order)
        acc_cons_tp = self.get_vel_matrix(times[-1],order)
        jerk_cons_tp = self.get_jerk_matrix(times[-1],order)
        matrix = np.vstack((loc_cons_t, loc_cons_tp, jerk_cons_tp))
        
        M_mat.append(self.constraints_row([matrix], [segs - 1], segs))

        b_mat.append(np.array([points[segs-1], points[0], 0]))
        #b_mat.append(np.array([points[segs-1], points[segs], 0,0]))

        vel_cons_tp = np.expand_dims(self.get_vel_matrix(times[-2],order),0)
        acc_cons_tp = np.expand_dims(self.get_acc_matrix(times[-2],order),0)

        jerk_cons_tp = np.expand_dims(self.get_jerk_matrix(times[-2],order),0)

        # Adding more self.constraints_row to make system fully determinable
        # M_mat.append(self.constraints_row([np.expand_dims(vel_cons_t,0),-vel_cons_tp], [segs - 1, segs - 2], segs))

        # b_mat.append(np.zeros(1))

        # M_mat.append(self.constraints_row([np.expand_dims(acc_cons_t,0),-acc_cons_tp], [segs - 1, segs - 2], segs))
        
        # b_mat.append(np.zeros(1))

        # M_mat.append(self.constraints_row([np.expand_dims(jerk_cons_t,0),-jerk_cons_tp], [segs - 1, segs - 2], segs))
    
        # b_mat.append(np.zeros(1))
        M = np.concatenate(M_mat, axis=0)
        b = np.concatenate(b_mat)
        

        constraints_mat.append({'type': 'eq', 'fun': lambda c: b - M @ c})

        # M_ineq = np.concatenate(M_ineq_mat,axis=0)
        # b_ineq = np.concatenate(b_ineq_mat)

        # constraints_mat.append({'type': 'ineq', 'fun': lambda c: b_ineq - M_ineq @ c})

        # Run optimization problem
        print("Optimising")
        #before = time.time()
        solution = opt.minimize(
            self.objective,
            coeffs,
            args=(times, order),
            constraints=constraints_mat,
            options={
                'maxiter': 10000,
                'disp': True
            })

        #after = time.time()
        #print("----- Optimization finished: {} secs.".format(after - before))
        if not solution['success']:
            print("Optimization failed: {}".format(solution['message']))
            return None

        # parameters of the entire trajectory
        return solution['x']


    def store_optimal_trajectory(self, file_name, data):
        '''
            Store the optimal trajectory in a csv file
        '''
        with open(file_name, 'w') as f:
            # store the coefficients in a json file
            json.dump(data, f)
        

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', type=str, default='waypoints.csv')
    parser.add_argument('-o', '--output', type=str, default='optimal_trajectory.json')
    parser.add_argument('-d', '--order', type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    optimizer = TrajectoryOptimizer(args.file, args.order)
    goals, times = optimizer.load_tajectory(args.file)
    coeffs = optimizer.optimize_trajectory(goals, times)
    if coeffs is not None:
        optimizer.store_optimal_trajectory(args.output, coeffs)
