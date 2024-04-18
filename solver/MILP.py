import time

import torch
import numpy as np
from tqdm import tqdm

from gurobipy import GRB
import gurobipy as gp

from solver.gurobi_setup import get_optimal_grb_model


'''
Compute RBPOA in H-representation

Parameters:
model           : torch model to verify
Hx + d <= 0     : collision set (avoid set)
cs              : facets to bound the resulting convex polytope
state_space_lbs : lower bound of the state space
state_space_ubs : upper bound of the state space
dist_max        : backreachable distance

return:
A ( x_j - x_i ) + B <= 0   :  RBPOA in H-representation
'''
def compute_RBPOA(
    model, H, d, cs,
    state_space_lbs, state_space_ubs,
    agent_1_id=0, agent_2_id=1,
    uncertainty=0.5
):
    
    start = time.time()

    start_index_1 = agent_1_id * 4
    start_index_2 = agent_2_id * 4

    m, _, _, clean_input = get_optimal_grb_model(model, H, d, state_space_lbs, state_space_ubs, uncertainty)

    A, B = [], []

    for c in tqdm(cs):

        c = torch.where(torch.abs(c) > 1e-7, c, 1e-7)
        minb, maxb = None, None

        # lower bound
        m.setObjective(
            c[0] * (clean_input[start_index_2]-clean_input[start_index_1]) + \
                c[1] * (clean_input[start_index_2+1] - clean_input[start_index_1+1]),
            GRB.MINIMIZE
        )
        m.optimize()
        if m.status == GRB.OPTIMAL:
            minb = m.getObjective().getValue()

        # upper bound
        m.setObjective(
            -c[0] * (clean_input[start_index_2]-clean_input[start_index_1]) - \
                c[1] * (clean_input[start_index_2+1] - clean_input[start_index_1+1]),
            GRB.MINIMIZE
        )
        m.optimize()
        if m.status == GRB.OPTIMAL:
            maxb = m.getObjective().getValue()
        
        A.append(c.cpu().numpy())
        A.append(-c.cpu().numpy())
        B.append(-minb)
        B.append(-maxb)
    
    A = np.array(A)
    B = np.array(B)
    m.dispose()
    gp.disposeDefaultEnv()
    
    return A, B, time.time()-start