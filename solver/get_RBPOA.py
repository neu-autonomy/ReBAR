import torch
import numpy as np

from solver.MILP import compute_RBPOA
from solver.sample_RBPUA import sample_RBPUA


'''
Returns the RBPOA for specified agents for specified number of steps, 
with specified uncertainty in state estimation.

Parameters:
model           : torch model to verify
Hx + d <= 0     : collision set (avoid set)
cs              : facets to bound the resulting convex polytope
state_space_lbs : lower bound of the state space
state_space_ubs : upper bound of the state space
dist_max        : backreachable distance
agent_1_id      : id of first agent to verify
agent_2_id      : id of second agent to verify
steps           : compute n-step RBPOA
uncertainty     : uncertainty in coordinate measurements
monte_carlo     : if set to True, return a polytope representing RBPUA from monte carlo sampling

Returns:
A ( x_j - x_i ) + B <= 0   :  RBPOA in H-representation
Au ( x_j - x_i ) + Bu <= 0 : RBPUA in H-representation
'''
def get_RBPOA(
    model,
    H, d,
    cs, 
    state_space_lbs,
    state_space_ubs,
    dist_max = 3.0,
    num_agents = 2,
    agent_1_id = 0,
    agent_2_id = 1,
    steps = 1,
    uncertainty = 0.5,
    monte_carlo = False,
    initial_state = None
):
    As, Bs, Aus, Bus = [], [], [], []
    d_original = d
    subtraction_matrix = H[:2]

    total_runtime = 0.0

    for step in range(steps):

        if step == 0 and initial_state is not None:
            d = d_original - H @ initial_state

        elif step > 0:

            H = torch.tensor(A).float() @ subtraction_matrix
            d = -torch.tensor(B).float()

        A, B, runtime = compute_RBPOA(
            model, H, d, cs,
            state_space_lbs, state_space_ubs,
            agent_1_id, agent_2_id,
            uncertainty
        )

        total_runtime += runtime

        As.append(A)
        Bs.append(B)

        # if monte_carlo:

        #     Au, Bu = sample_RBPUA(
        #         model, H, d,
        #         state_space_lbs, state_space_ubs,
        #         num_agents,
        #         agent_1_id, agent_2_id,
        #         dist_max+2*step,
        #         uncertainty,
        #         num_samples = 50000
        #     )

        #     Aus.append(Au)
        #     Bus.append(Bu)

    return As, Bs, Aus, Bus, total_runtime / steps