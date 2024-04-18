import torch
import torch.nn as nn

from model.model_setup_utils import construct_model_full_step, merge_controllers_multi_agents
from model.model import STATE_DIM, POLICY_DIM
from model.model import two_agent_orig, two_agent_nonres_ulimits, two_agent_nonres_ulimits_merged


def init_model_param_all_zeros(model):
    for layer in model:
        if isinstance(layer, nn.Linear):
            layer.weight = nn.Parameter(torch.zeros_like(layer.weight))
            layer.bias = nn.Parameter(torch.zeros_like(layer.bias))
    return model


def init_layer(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


'''
Setup the specified controllers by
1: incorporating control limits and system dynamics into ReLU MLP
2: merging two processed ReLU MLPs into a larger ReLU MLP
Verification will be run on the merged model

parameters:
A1, A2 (torch tensor):      state transition matrices for agent 1/2
B1, B2 (torch tensor):      control matrices for agent 1/2
path_orig_model_1 (str):    path to agent 1 NNC checkpoint
path_orig_model_2 (str):    path to agent 2 NNC checkpoint
u_lb, u_ub (torch tensor):  control signal vector lower and upper bound

returns:
merged network ready for verification
'''
def setup_model(
    A1, A2, B1, B2,
    path_orig_model_1, path_orig_model_2,
    hidden_dim_1=10, hidden_dim_2=10,
    n=2,
    u_lb = torch.Tensor([-1.0, -1.0]).cuda(),
    u_ub = torch.Tensor([1.0, 1.0]).cuda()
):

    controller_1 = two_agent_orig(hidden_dim_1, hidden_dim_2, n)
    controller_2 = two_agent_orig(hidden_dim_1, hidden_dim_2, n)

    if path_orig_model_1 != None:
        controller_1.load_state_dict(torch.load(path_orig_model_1, map_location=torch.device('cpu')))
    else:
        controller_1.apply(init_layer)

    if path_orig_model_2 != None:
        controller_2.load_state_dict(torch.load(path_orig_model_2, map_location=torch.device('cpu')))
    else:
        controller_2.apply(init_layer)

    full_step_1 = two_agent_nonres_ulimits(hidden_dim_1, hidden_dim_2, n)
    full_step_2 = two_agent_nonres_ulimits(hidden_dim_1, hidden_dim_2, n)

    full_step_1 = init_model_param_all_zeros(full_step_1)
    full_step_2 = init_model_param_all_zeros(full_step_2)

    # construct full dynamic network for agent 1
    full_step_1 = construct_model_full_step(
        controller_1, full_step_1, u_lb, u_ub, A1, B1, 
        STATE_DIM, hidden_dim_1, hidden_dim_2, POLICY_DIM, n)

    # construct full dynamic network for agent 2
    full_step_2 = construct_model_full_step(
        controller_2, full_step_2, u_lb, u_ub, A2, B2, 
        STATE_DIM, hidden_dim_1, hidden_dim_2, POLICY_DIM, n)
        
    merged_model = two_agent_nonres_ulimits_merged(hidden_dim_1, hidden_dim_2, n)
    merged_model = init_model_param_all_zeros(merged_model)
    
    merged_model = merge_controllers_multi_agents(full_step_1, full_step_2, merged_model, STATE_DIM, hidden_dim_1, hidden_dim_2, POLICY_DIM, n)

    return merged_model, controller_1, controller_2