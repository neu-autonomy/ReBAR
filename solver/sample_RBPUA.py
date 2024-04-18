import torch
import numpy as np

from solver.sampling_utils import sample_local_coordinate
from solver.polytope_utils import compute_RBPUA_Hrep


def sample_RBPUA(
    model, H, d,
    state_space_lbs, state_space_ubs,
    num_agents,
    agent_1_id, agent_2_id,
    dist_max,
    uncertainty = 0.5,
    num_samples = 200000,
):
    
    test_input = sample_local_coordinate(
        state_space_lbs, state_space_ubs,
        dist_max,
        agent_1_id, agent_2_id,
        num_samples
    )

    test_input = test_input.float().cuda()

    # add uncertainty to true state to generate measured noisy state
    # these states will be used as input to the controller
    low=[-uncertainty, -uncertainty, 0, 0]*num_agents
    high=[uncertainty, uncertainty, 0, 0]*num_agents
    uncertainty_ball = np.random.uniform(
        low=low, high=high,
        size=(test_input.shape[0], test_input.shape[1])
    )
    uncertainty_ball = torch.tensor(uncertainty_ball).float().cuda()
    controll_input = test_input + uncertainty_ball

    # only use noisy states that are within the state space
    index = (controll_input[:,0] >= state_space_lbs[0]) & (controll_input[:,0] <= state_space_ubs[0])
    for i in range(1, controll_input.shape[-1]):
        in_lower_bound = controll_input[:,i] >= state_space_lbs[i]
        in_upper_bound = controll_input[:,i] <= state_space_ubs[i]
        index = index & in_lower_bound & in_upper_bound
    controll_input = controll_input[index.nonzero()].view(-1, controll_input.shape[-1])
    test_input = test_input[index.nonzero()].view(-1, test_input.shape[-1])
    diff = controll_input - test_input

    # compute next step state
    test_output = model(controll_input)
    test_output -= torch.hstack([diff[:, agent_1_id*4:(agent_1_id+1)*4], diff[:, agent_2_id*4:(agent_2_id+1)*4]])

    # only keep input-output pairs that are within the state space
    index = (test_output[:,0] >= state_space_lbs[0]) & (test_output[:,0] <= state_space_ubs[0])
    for i in range(1, 8):
        in_lower_bound = test_output[:,i] >= state_space_lbs[i]
        in_upper_bound = test_output[:,i] <= state_space_ubs[i]
        index = index & in_lower_bound & in_upper_bound
    test_output = test_output[index.nonzero()].view(-1, test_output.shape[-1])
    test_input = test_input[index.nonzero()].view(-1, test_input.shape[-1])

    # get input that leads to collision
    Linf_dist_layer = torch.nn.Linear(8, 4)
    Linf_dist_layer.weight = torch.nn.Parameter(H)
    Linf_dist_layer.bias = torch.nn.Parameter(d)
    Linf_dist = Linf_dist_layer(test_output)
    Linf_dist = torch.max(Linf_dist, dim=1).values
    collide_indices = Linf_dist <= 0
    collide_indices = collide_indices.nonzero()
    collide_input = test_input[collide_indices]
    collide_input = collide_input.view(collide_input.shape[0], -1)
    collide_input = torch.hstack(
        [
            collide_input[:,(agent_1_id*4):(agent_1_id*4+2)], 
            collide_input[:,(agent_2_id*4):(agent_2_id*4+2)]
        ]
    )

    A, B = compute_RBPUA_Hrep(collide_input)

    return A, B