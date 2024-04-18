import torch
import numpy as np


def sample_local_coordinate(
    state_space_lbs, state_space_ubs, 
    dist_max=3., 
    agent_1_id=0, agent_2_id=1,
    amt=10000
):
    
    start_index_1 = agent_1_id * 4
    start_index_2 = agent_2_id * 4

    sample_lbs_ = np.array(state_space_lbs)
    sample_ubs_ = np.array(state_space_ubs)

    sample_lbs = np.hstack([sample_lbs_[start_index_1:(start_index_1+4)], sample_lbs_[start_index_2:(start_index_2+4)]])
    sample_ubs = np.hstack([sample_ubs_[start_index_1:(start_index_1+4)], sample_ubs_[start_index_2:(start_index_2+4)]])

    test_input = np.random.uniform(low=sample_lbs, high=sample_ubs, size=(amt, 8))

    test_input = np.hstack([test_input[:,:4], test_input[:,:2], test_input[:,-2:]])
    
    perturbation_lb = [0, 0, 0, 0, 0, 0, 0, 0]
    perturbation_ub = [0, 0, 0, 0, 0, 0, 0, 0]

    perturbation_lb[4] = -dist_max
    perturbation_lb[5] = -dist_max
    perturbation_ub[4] = dist_max
    perturbation_ub[5] = dist_max

    perturbation = np.random.uniform(low=perturbation_lb, high=perturbation_ub, size=(amt, 8))

    test_input = test_input + perturbation
    test_input = torch.tensor(test_input).float()

    left_in_bound = test_input[:,4] >= state_space_lbs[0]
    right_in_bound = test_input[:,4] <= state_space_ubs[0]
    bottom_in_bound = test_input[:,5] >= state_space_lbs[1]
    top_in_bound = test_input[:,5] <= state_space_ubs[1]

    in_bound = left_in_bound & right_in_bound & bottom_in_bound & top_in_bound
    in_bound = in_bound.nonzero()
    test_input = test_input[in_bound]
    test_input = test_input.view(test_input.shape[0], -1)
    
    test_input_base = np.random.uniform(low=state_space_lbs, high=state_space_ubs, size=(test_input.shape[0], len(state_space_lbs)))
    test_input_base = torch.tensor(test_input_base)
    test_input_base[:,start_index_1:(start_index_1+4)] = test_input[:, :4]
    test_input_base[:,start_index_2:(start_index_2+4)] = test_input[:, -4:]

    test_input_base = test_input_base.view(test_input_base.shape[0], -1)

    return test_input_base


def sample_online_unsafe_region(box, amt=1000):

    test_input = np.random.uniform(
        low=[box[0], box[1], -1, -1, box[2], box[3], -1, -1],
        high=[box[4], box[5], 1, 1, box[6], box[7], 1, 1],
        size=(amt, 8)
    )
    test_input = torch.tensor(test_input).float().cuda()
    return test_input