import argparse
import glob
import matplotlib.pyplot as plt

import torch
import numpy as np

from model.model import STATE_DIM, POLICY_DIM
from model.model_setup_for_verification import setup_model

from solver.get_RBPOA import get_RBPOA
from utils.visualization_utils import plot_polytope, plot_util_sets


def parse_arguments():

    parser = argparse.ArgumentParser(description="n-agent single integrator potential field policy RBPOA")
    parser.add_argument('--lb_x', type=float, default=-10., help="input lower bound of agents x coord")
    parser.add_argument('--ub_x', type=float, default=10., help="input upper bound of agents x coord")
    parser.add_argument('--lb_y', type=float, default=-10., help="input lower bound of agents y coord")
    parser.add_argument('--ub_y', type=float, default=10., help="input upper bound of agents y coord")
    parser.add_argument('-r', type=float, default=1., help="Minimum Safety Radius Between Agents")
    parser.add_argument('--state_uncertainty', type=float, default=0.5, help="size of L-inf epsilon ball around uncertain state we consider")
    parser.add_argument('--num_cs', type=int, default=20, help="number of half-planes for backprojection set.")
    parser.add_argument('--num_agents', type=int, default=5, help="number of dummy agents in the system.")
    parser.add_argument('--num_exps', type=int, default=1, help="number of experiments to run (and average).")
    parser.add_argument('--hidden_dim_1', type=int, default=20, help="first hidden dim of controller")
    parser.add_argument('--hidden_dim_2', type=int, default=20, help='second hidden dim of controller')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="directory containing the checkpoint files")
    parser.add_argument('--RBPUA', action="store_true", help="whether to sample RBPUA or not")
    parser.add_argument('--plot', action='store_true', help="whether to plot results or not")

    args = parser.parse_args()

    return args


def generate_A_matrix(agent_num, total_agents=3):

    A = torch.zeros((4, 4*total_agents))
    A[0, agent_num*4] = 1.0
    A[1, agent_num*4+1] = 1.0
    return A


if __name__ == '__main__':

    args = parse_arguments()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.set_default_device('cuda:0')
        torch.manual_seed(1)

    # defines the matrices for computing pairwise distances
    H = torch.zeros((4, 2*STATE_DIM+2*POLICY_DIM))
    H[:2, :STATE_DIM] = torch.eye(STATE_DIM)
    H[:2, STATE_DIM+POLICY_DIM:2*STATE_DIM+POLICY_DIM] = -torch.eye(STATE_DIM)
    H[2:, :STATE_DIM] = -torch.eye(STATE_DIM)
    H[2:, STATE_DIM+POLICY_DIM:2*STATE_DIM+POLICY_DIM] = torch.eye(STATE_DIM)

    # matrix of minimum safety radius
    r = args.r
    d = -torch.Tensor([r, r, r, r])

    # input matrices
    B = torch.Tensor(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1]
        ]
    )

    state_space_lbs = [args.lb_x, args.lb_y, -1, -1] * args.num_agents
    state_space_ubs = [args.ub_x, args.ub_y, 1, 1] * args.num_agents

    # define the slopes of the bounding halfplanes
    num_cs = args.num_cs
    cs = [[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)]
    cs = torch.tensor(cs)

    dist_max = r + 2

    # grab all agents' controller checkpoints
    checkpoints = []
    if args.checkpoint_dir:
        checkpoints = glob.glob(args.checkpoint_dir+'/*')
        checkpoints = sorted(checkpoints)
        if len(checkpoints) == 10:
            checkpoints = [checkpoints[0]] + checkpoints[2:] + [checkpoints[1]]

    runtimes = []

    for e in range(args.num_exps):

        print("===========================================")
        print(f"+              Experiment {e}               +")
        print("===========================================")

        for i in range(args.num_agents):
            for j in range(i+1, args.num_agents):

                Ai = generate_A_matrix(i, args.num_agents)
                Aj = generate_A_matrix(j, args.num_agents)

                ckpt1 = None if len(checkpoints) == 0 else checkpoints[i]
                ckpt2 = None if len(checkpoints) == 0 else checkpoints[j]

                modelij, controller_i, controller_j = setup_model(
                    Ai, Aj, B, B,
                    ckpt1, ckpt2,
                    hidden_dim_1=args.hidden_dim_1,
                    hidden_dim_2=args.hidden_dim_2,
                    n=args.num_agents
                )

                As, Bs, Aus, Bus, runtime = get_RBPOA(
                    model = modelij,
                    H = H, 
                    d = d, 
                    cs = cs,
                    state_space_lbs = state_space_lbs, 
                    state_space_ubs = state_space_ubs,
                    dist_max = dist_max, 
                    num_agents = args.num_agents,
                    agent_1_id = i, 
                    agent_2_id = j, 
                    uncertainty = args.state_uncertainty,
                    monte_carlo = args.RBPUA
                )

                runtimes.append(runtime)

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

    print(f"Average runtime per pair for {args.num_agents} agents is {sum(runtimes) / len(runtimes)}s")