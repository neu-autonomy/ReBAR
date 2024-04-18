from typing import List

import torch
import numpy as np
import gurobipy as gp
from gurobipy import gurobipy


def get_num_layers(model: torch.nn.Sequential):
    layers = len(model) // 2
    assert layers * 2 == len(model), "Model should have an even number of entries"
    return layers


def _get_grb_model(
        model: torch.nn.Sequential,
        layers: int,
        input_lbs: List[float],
        input_ubs: List[float],
        uncertainty=1):
    
    m = gp.Model("verify_input")
    m.Params.OutputFlag = 0
 
    # Create variables
    clean_input = m.addMVar(shape=len(input_lbs), lb=input_lbs, ub=input_ubs, name="clean_input")

    # add uncertainty to controller input
    uncertainty_ball = np.zeros(len(input_lbs))
    for i in range(len(input_lbs)):
        if i % 4 == 0 or i % 4 == 1:
            uncertainty_ball[i] = uncertainty
    epsilon_ball = m.addMVar(shape=len(input_lbs), lb=-uncertainty_ball, ub=uncertainty_ball, name="uncertainty")
    input = m.addMVar(shape=len(input_lbs), lb=input_lbs, ub=input_ubs, name="input")
    m.addConstr(input == (clean_input + epsilon_ball))
    diff = m.addMVar(shape=len(input_lbs))
    m.addConstr(diff == (input - clean_input))

    xs = []
    zs = [input]
    for l in range(layers-1):
        w = model[l*2 + 1].weight.detach().cpu().numpy()
        hidden_dim = w.shape[0]
        xs.append(m.addMVar(shape=hidden_dim, lb=-1e30, ub=1e30, name=f"x{l}"))
        zs.append(m.addMVar(shape=hidden_dim, lb=-1e30, ub=1e30, name=f"z{l+1}"))
    output_dim = model[-1].weight.shape[0]
    xs.append(m.addMVar(shape=output_dim, lb=input_lbs[:8], ub=input_ubs[:8], name="output"))
    # uncomment this when doing bifurcating policy
    # xs.append(m.addMVar(shape=output_dim, lb=-1e30, ub=1e30, name="output"))
    return m, xs, zs, diff, clean_input


def get_optimal_grb_model(
    model: torch.nn.Sequential,
    H: torch.Tensor,
    d: torch.Tensor,
    input_lbs: List[float],
    input_ubs: List[float],
    uncertainty=1
):
    
    layers = get_num_layers(model)
 
    m, xs, zs, epsilon_ball, clean_input = _get_grb_model(model, layers, input_lbs, input_ubs, uncertainty)

    for layer in range(layers-1):

        w = model[layer*2 + 1].weight.detach().cpu().numpy()
        b = model[layer*2 + 1].bias.detach().cpu().numpy()
        hidden_dim = w.shape[0]

        if layer != 0:

            m.addConstr(((w @ zs[layer]) + b) == xs[layer])

        else:
            # recover "physical" or true state for dynamic step
            start_index_1 = int(hidden_dim/2 - len(input_lbs))
            start_index_2 = int(hidden_dim - len(input_lbs))
            first_linear_output = m.addMVar(shape=hidden_dim, lb=-1e30, ub=1e30)
            m.addConstr(first_linear_output == ((w @ zs[layer]) + b))
            m.addConstr(xs[layer][:start_index_1] == first_linear_output[:start_index_1])
            m.addConstr(
                xs[layer][start_index_1:int(hidden_dim/2)] == \
                    (first_linear_output[start_index_1:int(hidden_dim/2)]-epsilon_ball))
            m.addConstr(xs[layer][int(hidden_dim/2):start_index_2] == first_linear_output[int(hidden_dim/2):start_index_2])
            m.addConstr(xs[layer][start_index_2:] == (first_linear_output[start_index_2:]-epsilon_ball))
                
        for i in range(hidden_dim):
            m.addConstr(zs[layer+1][i] == gp.max_(xs[layer][i], constant=0))

    w = model[-1].weight.detach().cpu().numpy()
    b = model[-1].bias.detach().cpu().numpy()
    m.addConstr(((w @ zs[-1]) + b) == xs[-1])

    if H is not None and d is not None:

        m.addConstr(H.detach().cpu().numpy() @ xs[-1] + d.detach().cpu().numpy() <= 0)
 
    return m, xs, zs, clean_input