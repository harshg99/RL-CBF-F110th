"""Verify a CBF by sampling"""
from typing import cast, List, Tuple, Optional, TYPE_CHECKING
from itertools import product

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import seaborn as sns
import torch
import tqdm

class CBFVerificationExperiment():
    """An experiment for verifying learned CBFs on a grid.

    WARNING: VERY SLOW. Exponential!!
    """

    def __init__(
        self,
        domain: Optional[List[Tuple[float, float]]] = None,
        n_grid: int = 1000,
    ):
        """Initialize an experiment for validating the CBF over a given domain.

        args:
            name: the name of this experiment
            domain: a list of two tuples specifying the plotting range,
                    one for each state dimension.
            n_grid: the number of points in each direction at which to compute V
        """

        # Default to plotting over [-1, 1] in all directions
        if domain is None:
            domain = [(-14.0, -14.0), (14.0, 14.0)]
        self.domain = domain

        self.n_grid = n_grid
        self.plot_unsafe_region = True
        self.x_axis_label = "0"
        self.y_axis_label = "1"

    @torch.no_grad()
    def run(self, controller_under_test) -> pd.DataFrame:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Set up a dataframe to store the results
        results = []

        # We need a vector for each dimension
        n_dims = controller_under_test.dynamics_model.n_dims
        state_vals = torch.zeros(n_dims, self.n_grid)
        for dim_idx in range(n_dims):
            state_vals[dim_idx, :] = torch.linspace(
                self.domain[dim_idx][0],
                self.domain[dim_idx][1],
                self.n_grid,
                device=device,
            )

        # Loop through the grid, which is a bit tricky since we have to loop through
        # all dimensions, and we don't know how many we have right now. We'll use
        # a cartesian list product to get all points in the grid.
        prog_bar = tqdm.tqdm(product(*state_vals), desc="Validating CBF", leave=True)
        for point in prog_bar:
            x = torch.tensor(point).view(1, -1)

            # Get the value and the Lie derivative of CBF
            V, JV = controller_under_test.V_with_jacobian(x)
            Lf_V = (JV * controller_under_test.dynamic_system.closed_loop_dynamics(x,controller_under_test.policy_net(x))).sum(axis=1)
            lie_derivative = Lf_V + controller_under_test.cbf_lambda*V.squeeze()

            # Get the goal, safe, or unsafe classification
            is_goal = controller_under_test.dynamic_system.goal_mask(x).all()
            is_safe = controller_under_test.dynamic_system.safe_mask(x).all()
            is_unsafe = controller_under_test.dynamic_system.unsafe_mask(x).all()

            # Store the results
            log_packet = {
                "V": V.cpu().numpy().item(),
                "Goal region": is_goal.cpu().numpy().item(),
                "Safe region": is_safe.cpu().numpy().item(),
                "Unsafe region": is_unsafe.cpu().numpy().item(),
                "Lie derivative": lie_derivative.cpu().numpy().item()
            }
            state_list = x.squeeze().cpu().tolist()
            for state_idx, state in enumerate(state_list):
                log_packet[str(state_idx)] = state

            results.append(log_packet)
        return pd.DataFrame(results)

    def plot(
        self,
        controller_under_test: "Controller",
        results_df: pd.DataFrame,
        display_plots: bool = False,
    ) -> List[Tuple[str, figure]]:
        """
        Plot the results, and return the plot handles. Optionally
        display the plots.

        args:
            controller_under_test: the controller with which to run the experiment
            display_plots: defaults to False. If True, display the plots (blocks until
                           the user responds).
        returns: a list of tuples containing the name of each figure and the figure
                 object.
        """
        # Set the color scheme
        sns.set_theme(context="talk", style="white")

        # Plot a contour of V
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 8)
        contours = ax.tricontourf(
            results_df["0"],
            results_df["1"],
            results_df["V"],
            levels=20,
        )
        plt.colorbar(contours, ax=ax, orientation="vertical")
       
        # Filter the dataframe by Lie derivative greater than 1e-5
        import pdb
        pdb.set_trace()
        filtered_df = results_df[results_df["Lie derivative"] > 1e-5]
        
        # Plot the filtered dataframe as a scatter plot
        ax.scatter(filtered_df[self.x_axis_label], filtered_df[self.y_axis_label], c="red", label="Descent condition violation", s=5, marker="o")
        
        
        if self.plot_unsafe_region:
            ax.plot([], [], c="green", label="Safe Region")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Safe region"],
                colors=["green"],
                levels=[0.5],
            )
            ax.plot([], [], c="magenta", label="Unsafe Region")
            ax.tricontour(
                results_df[self.x_axis_label],
                results_df[self.y_axis_label],
                results_df["Unsafe region"],
                colors=["magenta"],
                levels=[0.5],
            )
            ax.plot([], [], c="blue", label="V(x) = c")
            if hasattr(controller_under_test, "safe_level"):
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["V"],
                    colors=["blue"],
                    levels=[controller_under_test.safe_level], 
                )
            else:
                ax.tricontour(
                    results_df[self.x_axis_label],
                    results_df[self.y_axis_label],
                    results_df["V"],
                    colors=["blue"],
                    levels=[0.0],
                )
            ax.legend(loc="upper left")
        plt.savefig("my_plot.png")
