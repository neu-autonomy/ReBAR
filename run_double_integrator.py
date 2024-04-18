import argparse
import matplotlib.pyplot as plt

import torch
import numpy as np

from model.model import STATE_DIM, POLICY_DIM
from model.model_setup_for_verification import setup_model
from solver.get_RBPOA import get_RBPOA
from utils.visualization_utils import plot_polytope, plot_util_sets


def generate_A_matrix(agent_num):

    A = torch.zeros((4, 8))
    A[:, (agent_num*4):((agent_num+1)*4)] = torch.eye(4)
    A[:2, (agent_num*4+2):((agent_num+1)*4)] = torch.eye(2)
    return A


def parse_arguments():

    parser = argparse.ArgumentParser(description="2 Agent double integrator RVO Policy RBPOA")
    parser.add_argument('--lb_x', type=float, default=0, help="input lower bound of agents x coord")
    parser.add_argument('--ub_x', type=float, default=10., help="input upper bound of agents x coord")
    parser.add_argument('--lb_y', type=float, default=0, help="input lower bound of agents y coord")
    parser.add_argument('--ub_y', type=float, default=10., help="input upper bound of agents y coord")
    parser.add_argument('--hidden_dim_1', type=int, default=10, help="first hidden dim of controller")
    parser.add_argument('--hidden_dim_2', type=int, default=10, help='second hidden dim of controller')
    parser.add_argument('--checkpoint_1', type=str, default=None, help='if model_init == load, init agent 1 weights from this checkpoint file')
    parser.add_argument('--checkpoint_2', type=str, default=None, help='if model_init == load, init agent 2 weights from this checkpoint file')
    parser.add_argument('--state_uncertainty', type=float, default=0.5, help="size of L-inf epsilon ball around uncertain state we consider")
    parser.add_argument('-r', type=float, default=1., help="Minimum Safety Radius Between Agents")
    parser.add_argument('--num_cs', type=int, default=20, help="number of half-planes for backprojection set.")
    parser.add_argument('--RBPUA', action="store_true", help="whether to sample RBPUA or not")
    parser.add_argument('--plot', action='store_true', help="whether to plot results or not")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_arguments()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_device('cuda:0')

    # defines the matrices for computing pairwise distances
    H = torch.zeros((4, 2*STATE_DIM+2*POLICY_DIM))
    H[:2, :STATE_DIM] = torch.eye(STATE_DIM)
    H[:2, STATE_DIM+POLICY_DIM:2*STATE_DIM+POLICY_DIM] = -torch.eye(STATE_DIM)
    H[2:, :STATE_DIM] = -torch.eye(STATE_DIM)
    H[2:, STATE_DIM+POLICY_DIM:2*STATE_DIM+POLICY_DIM] = torch.eye(STATE_DIM)

    # matrix of minimum safety radius
    r = args.r
    d = -torch.Tensor([r, r, r, r])
    dist_max = r + 2

    A1 = generate_A_matrix(0)
    A2 = generate_A_matrix(1)

    B = torch.Tensor(
        [
            [0.5, 0.],
            [0., 0.5],
            [1., 0.],
            [0., 1.]
        ]
    )

    accel_lb = torch.tensor([-2., -2.]).cuda()
    accel_ub = torch.tensor([2., 2.]).cuda()

    state_space_lbs = [args.lb_x, args.lb_y, -1, -1, args.lb_x, args.lb_y, -1, -1]
    state_space_ubs = [args.ub_x, args.ub_y, 1, 1, args.ub_x, args.ub_y, 1, 1]

    num_cs = args.num_cs
    cs = [[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)]
    cs = torch.tensor(cs)

    ckpt1 = args.checkpoint_1
    ckpt2 = args.checkpoint_2

    model, _, _ = setup_model(
        A1, A2, B, B,
        ckpt1, ckpt2,
        hidden_dim_1 = args.hidden_dim_1,
        hidden_dim_2 = args.hidden_dim_2,
        n = 2,
        u_lb = accel_lb,
        u_ub = accel_ub
    )

    As, Bs, Aus, Bus, runtime = get_RBPOA(
        model = model,
        H = H,
        d = d,
        cs = cs,
        state_space_lbs = state_space_lbs,
        state_space_ubs = state_space_ubs,
        dist_max = dist_max,
        num_agents = 2,
        agent_1_id = 0,
        agent_2_id = 1,
        uncertainty = args.state_uncertainty,
        monte_carlo = args.RBPUA
    )

    if args.plot:

        # intialize matplotlib figure
        plt.figure(0)
        plt.rcParams.update({'font.size': 50})
        axis = plt.gca()

        plot_polytope(As[0], Bs[0], 0, axis)
        plot_polytope(-Aus[0], Bus[0], 0, axis, color='blue')
        plot_util_sets(r, dist_max, 0, axis)

        plt.xlim(-dist_max-1, dist_max+1)
        plt.ylim(-dist_max-1, dist_max+1)

        plt.show()

        exit(0)