import torch
import torch.nn as nn


'''
Verify processed MLP does control projection and system dynamics correctly
'''
def verify_full_step(orig_model, full_step, u_lb, u_ub, A, B, input_dim):
    input_tensor = torch.randn((100, input_dim))
    policy_output = orig_model(input_tensor)
    policy_output = torch.clip(policy_output, min=u_lb, max=u_ub)
    policy_output = nn.functional.linear(input_tensor, A, bias=None) \
        + nn.functional.linear(policy_output, B, bias=None)
    full_step_output = full_step(input_tensor)
    assert (torch.abs(policy_output - full_step_output) < 1e-5).all()


'''
Incorporates control limit projection and system dynamics into the ReLU MLP

Parameters:
orig_model (torch module):     Loaded with pretrained weights. The policy NNC
full_step (torch module):      Empty weights. Will be filled up using weights in orig_model + dynamics
u_lb, u_ub (torch tensor)      control input lower and upper bound
A (torch tensor):              n x 2*n size tensor. State Matrix of the multi agent dynamic
B (torch tensor):              n x n_u size tensor. Control Matrix of the multi agent dynamic
STATE_DIM (int):               dimension of state vector
HIDDEN1_DIM (int):             dimension of nnc first hidden layer
HIDDEN2_DIM (int):             dimension of nnc second hidden layer
POLICY_DIM (int):              dimension of control vector
n (int):                            number of agents in the system

returns:
ReLU MLP with control limit projection and system dynamics
'''
def construct_model_full_step(
    orig_model, full_step, u_lb, u_ub, A, B, 
    STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM,
    n=2
):
    
    with torch.no_grad():

        full_step[1].weight[:HIDDEN1_DIM] = orig_model[1].weight
        full_step[1].weight[-(n*STATE_DIM+n*POLICY_DIM):] = nn.Parameter(torch.eye(n*STATE_DIM+n*POLICY_DIM))
        full_step[1].bias[:HIDDEN1_DIM] = orig_model[1].bias
        full_step[1].bias[-(n*STATE_DIM+n*POLICY_DIM):] = \
            nn.Parameter(torch.Tensor([10.0 for _ in range(n*STATE_DIM+n*POLICY_DIM)]))

        full_step[3].weight[:HIDDEN2_DIM, :HIDDEN1_DIM] = orig_model[3].weight
        full_step[3].weight[-(n*STATE_DIM+n*POLICY_DIM):, -(n*STATE_DIM+n*POLICY_DIM):] = \
            nn.Parameter(torch.eye(n*STATE_DIM+n*POLICY_DIM))
        full_step[3].bias[:HIDDEN2_DIM] = orig_model[3].bias

        full_step[5].weight[:POLICY_DIM, :HIDDEN2_DIM] = orig_model[5].weight
        full_step[5].weight[-(n*STATE_DIM+n*POLICY_DIM):, -(n*STATE_DIM+n*POLICY_DIM):] = \
            nn.Parameter(torch.eye(n*STATE_DIM+n*POLICY_DIM))
        full_step[5].bias[:POLICY_DIM] = orig_model[5].bias - u_lb

        full_step[7].weight[:POLICY_DIM, :POLICY_DIM] = nn.Parameter(-torch.eye(POLICY_DIM))
        full_step[7].weight[-(n*STATE_DIM+n*POLICY_DIM):, -(n*STATE_DIM+n*POLICY_DIM):] = \
            nn.Parameter(torch.eye(n*STATE_DIM+n*POLICY_DIM))
        full_step[7].bias[:POLICY_DIM] = nn.Parameter(torch.Tensor(u_ub - u_lb))

        full_step[9].weight[:, :POLICY_DIM] = nn.Parameter(-B)
        full_step[9].weight[:, -(n*STATE_DIM+n*POLICY_DIM):] = nn.Parameter(A)
        full_step[9].bias = nn.Parameter(
            u_ub.matmul(B.T) - A.matmul(torch.Tensor([10.0 for _ in range(n*STATE_DIM+n*POLICY_DIM)]))
        )

    verify_full_step(orig_model, full_step, u_lb, u_ub, A, B, n*STATE_DIM+n*POLICY_DIM)
    return full_step


'''
Verify merged model updates system correctly
'''
def verify_merged_controllers_multi_agents(merged_model, full_step_1, full_step_2, STATE_DIM, POLICY_DIM, n=3):
    state = []
    for i in range(n):
        state.append(torch.randn((100, STATE_DIM+POLICY_DIM)))
    input_tensor = torch.stack(state, dim=1).view(100, -1)
    output_tensor_1 = full_step_1(input_tensor)
    output_tensor_2 = full_step_2(input_tensor)
    merged_output = merged_model(input_tensor)
    full_step_output_merged = torch.stack([output_tensor_1, output_tensor_2], dim=1).view(100, -1)
    assert (torch.abs(full_step_output_merged - merged_output) < 1e-4).all()


'''
merge the two full step MLP into one. Note now the system has 3 agents

parameters:
full_step_1 (torch module):         agent 1 NNC with control limit projection and system dynamics
full_step_2 (torch module):         agent 2 NNC with control limit projection and system dynamics
merged_model (torch module):        the module to keep the merged network
STATE_DIM (int):                    dimension of state vector
HIDDEN1_DIM (int):                  dimension of nnc first hidden layer
HIDDEN2_DIM (int):                  dimension of nnc second hidden layer
POLICY_DIM (int):                   dimension of control vector

returns:
merged ReLU MLP representing the two agent system update of 1 discrete timestep
'''
def merge_controllers_multi_agents(
    full_step_1, full_step_2, merged_model, 
    STATE_DIM, HIDDEN1_DIM, HIDDEN2_DIM, POLICY_DIM, 
    n=3
):
    with torch.no_grad():

        weight_extend_states = torch.zeros((2*n*STATE_DIM+2*n*POLICY_DIM, n*STATE_DIM+n*POLICY_DIM))
        weight_layer_1 = torch.zeros((2 * (HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n), 2*n*STATE_DIM+2*n*POLICY_DIM))
        weight_extend_states[:n*STATE_DIM+n*POLICY_DIM, :] = torch.eye(n*STATE_DIM+n*POLICY_DIM)
        weight_extend_states[-(n*STATE_DIM+n*POLICY_DIM):, :] = torch.eye(n*STATE_DIM+n*POLICY_DIM)
        weight_layer_1[:(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n), :n*STATE_DIM+n*POLICY_DIM] = full_step_1[1].weight.data
        weight_layer_1[-(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n):, -(n*STATE_DIM+n*POLICY_DIM):] = full_step_2[1].weight.data
        merged_model[1].weight = nn.Parameter(weight_layer_1 @ weight_extend_states)
        merged_model[1].bias[:(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[1].bias
        merged_model[1].bias[-(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[1].bias

        merged_model[3].weight[:(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n), :(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[3].weight
        merged_model[3].weight[-(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n):, -(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[3].weight
        merged_model[3].bias[:(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[3].bias
        merged_model[3].bias[-(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[3].bias

        merged_model[5].weight[:(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n), :(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[5].weight
        merged_model[5].weight[-(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n):, -(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[5].weight
        merged_model[5].bias[:(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[5].bias
        merged_model[5].bias[-(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[5].bias

        merged_model[7].weight[:(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n), :(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[7].weight
        merged_model[7].weight[-(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n):, -(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[7].weight
        merged_model[7].bias[:(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[7].bias
        merged_model[7].bias[-(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[7].bias

        merged_model[9].weight[:(STATE_DIM + POLICY_DIM), :(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n)] = full_step_1[9].weight
        merged_model[9].weight[-(STATE_DIM + POLICY_DIM):, -(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n):] = full_step_2[9].weight
        merged_model[9].bias[:(STATE_DIM + POLICY_DIM)] = full_step_1[9].bias
        merged_model[9].bias[-(STATE_DIM + POLICY_DIM):] = full_step_2[9].bias
    
        verify_merged_controllers_multi_agents(merged_model, full_step_1, full_step_2, STATE_DIM, POLICY_DIM, n)

    return merged_model