import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import torch
import numpy as np
import random

from model.model import STATE_DIM, POLICY_DIM
from model.model_setup_for_verification import setup_model
from solver.get_RBPOA import get_RBPOA
from utils.visualization_utils import plot_polytope, plot_util_sets


def generate_A_matrix(agent_num, total_agents=2):

    A = torch.zeros((4, 4*total_agents))
    A[0, agent_num*4] = 1.0
    A[1, agent_num*4+1] = 1.0
    return A


def parse_arguments():

    parser = argparse.ArgumentParser(description="2 Agent single integrator RVO Policy RBPOA")
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
    parser.add_argument('--agent_1_start_x', type=float, default=5, help="agent 1 start x coord")
    parser.add_argument('--agent_1_start_y', type=float, default=9., help="agent 1 start y coord")
    parser.add_argument('--agent_2_start_x', type=float, default=9, help="agent 2 start x coord")
    parser.add_argument('--agent_2_start_y', type=float, default=5., help="agent 2 start y coord")
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

    # define the facets of the bounding halfplanes
    num_cs = args.num_cs
    cs = [[np.cos(2*np.pi*t / (num_cs*2)), np.sin(2*np.pi*t / (num_cs*2))] for t in range(num_cs)]
    cs = torch.tensor(cs)

    A1 = generate_A_matrix(0, 2)
    A2 = generate_A_matrix(1, 2)

    B = torch.vstack([torch.eye(2), torch.eye(2)])

    ckpt1 = args.checkpoint_1
    ckpt2 = args.checkpoint_2

    model_1, controller_1, controller_2 = setup_model(
        A1, A2, B, B,
        ckpt1, ckpt2,
        hidden_dim_1 = args.hidden_dim_1,
        hidden_dim_2 = args.hidden_dim_2,
        n = 2
    )

    model_2, _, _ = setup_model(
        A2, A1, B, B,
        ckpt2, ckpt1,
        hidden_dim_1 = args.hidden_dim_1,
        hidden_dim_2 = args.hidden_dim_2,
        n = 2
    )

    state_space_lbs = [args.lb_x, args.lb_y, -1, -1, args.lb_x, args.lb_y, -1, -1]
    state_space_ubs = [args.ub_x, args.ub_y, 1, 1, args.ub_x, args.ub_y, 1, 1]


    agent_to_RBPOA = dict()

    results_1 = get_RBPOA(
        model=model_1,
        H=H,
        d=d,
        cs=cs,
        agent_1_id=0,
        agent_2_id=1,
        state_space_lbs=state_space_lbs,
        state_space_ubs=state_space_ubs,
        dist_max=dist_max,
        steps=1,
        uncertainty=args.state_uncertainty,
        monte_carlo=False
    )
    results_2 = get_RBPOA(
        model=model_2,
        H=H,
        d=d,
        cs=cs,
        agent_1_id = 1,
        agent_2_id = 0,
        state_space_lbs=state_space_lbs,
        state_space_ubs=state_space_ubs,
        dist_max=dist_max,
        steps=1,
        uncertainty=args.state_uncertainty,
        monte_carlo=False
    )

    keys = ["As", "Bs", "Aus", "Bus", "runtime"]
    agent_to_RBPOA["1"] = {key: value for key, value in zip(keys, results_1)}
    agent_to_RBPOA["2"] = {key: value for key, value in zip(keys, results_2)}


    positions = [[args.agent_1_start_x, args.agent_1_start_y], [args.agent_2_start_x, args.agent_2_start_y]]
    velocities = [[0, 0], [0, 0]]
    agent_states = [[], []]

    # Initialize matplotlib figure
    plt.figure(0)
    plt.rcParams.update({'font.size': 50})
    axis = plt.gca()

    controllers = [controller_1, controller_2]

    # Loop for 10 iterations, possibly make this an arg? or find some better way to check if agents stop moving
    steps = 10
    for i in range(steps):
        prev_states = [positions[0][:], positions[1][:]]

        input_tensor = torch.tensor([[*positions[0], *velocities[0], *positions[1], *velocities[1]]], dtype=torch.float32)

        # Update positions for each agent
        for j in range(2):
            next_state = controllers[j](input_tensor)
            velocities[j] = [next_state[0][0].item(), next_state[0][1].item()]
            positions[j][0] += velocities[j][0]
            positions[j][1] += velocities[j][1]
            agent_states[j].append(positions[j][:]) 

        # Check if both agents stop moving
        if positions[0] == prev_states[0] and positions[1] == prev_states[1]:
            print(i)
            break
    
    def plot_agent(axis, position, agent_data, color):
        """Helper function to plot agent position and polytope."""
        axis.plot(position[0], position[1], 'o', color=color)
        plot_polytope(
            agent_data['As'][0], agent_data['Bs'][0],
            0, axis, color=color, offset=position
        )

    def create_animation():
        fig, axis = plt.subplots()
        plt.rcParams.update({'font.size': 50})

        def animate(i):
            axis.clear()
            bounds = (-5, 15) # Edit this for changing bounds
            axis.set_xlim(*bounds)
            axis.set_ylim(*bounds)

            plot_util_sets(r, dist_max, 0, axis)
            plot_agent(axis, agent_states[0][i], agent_to_RBPOA['1'], 'brown')
            plot_agent(axis, agent_states[1][i], agent_to_RBPOA['2'], 'olive')

        ani = animation.FuncAnimation(
            fig, animate, frames=len(agent_states[0]), interval=500, repeat=True
        )

        os.makedirs('./outputs', exist_ok=True)
        file_name = f'outputs/animation_{ckpt1.replace("/", ".")}_{ckpt2.replace("/", ".")}steps.gif'
        
        ani.save(file_name, writer='pillow', fps=2)  # apt install pillow
        plt.show()

    if args.plot:
        create_animation()
        

    

    plt.show()

    exit(0)

