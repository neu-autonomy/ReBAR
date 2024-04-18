import torch
import numpy as np

from model.train.RVO import RVO_update, compute_V_des


'''
Generate training samples for mimicing RVO policy
X: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
Y: [vx1', vy1'] and [vx2', vy2'] OR
   [ax1', ay1'] and [ax2', ay2']

Parameters:
state_space_lbs     : lower bound of state space
state_space_ubs     : upper bound of state space
control_limit       : max norm (per dimension) of the control signal (Y)
POLICY_DIM          : dimension of controlsignal vector
goal_1              : coordinate of goal of agent 1
goal_2              : coordinate of goal of agent 2
r                   : safety radius of the agents
num_samples         : number of training examples to generate
double_integ        : whether to generate acceleration as control or not. Default to FALSE
'''
def generate_sample(
    state_space_lbs, 
    state_space_ubs,
    control_limit,
    POLICY_DIM,
    goal_1,
    goal_2,
    r = 0.5,
    num_samples = 2000000,
    double_integ = False
):
    
    X = np.random.uniform(
        low=state_space_lbs, high=state_space_ubs, size=(num_samples, len(state_space_ubs))
    )

    V_max = control_limit

    goal = [goal_1, goal_2]

    #define workspace model
    ws_model = dict()
    #robot radius
    ws_model['robot_radius'] = r
    #circular obstacles, format [x,y,rad]
    # no obstacles
    ws_model['circular_obstacles'] = []
    #rectangular boundary, format [x,y,width/2,heigth/2]
    ws_model['boundary'] = []

    Y = []

    # Query RVO algorithm for optimal velocities
    for i in range(len(X)):
        if i % 100000 == 0:
            print(f"Finished generating {i} data pairs")
        x = np.vstack([X[i,:2], X[i,4:6]])
        v = np.vstack([X[i,2:4], X[i,-2:]])
        V_des = compute_V_des(x, goal, V_max)
        y = RVO_update(x, V_des, v, ws_model)
        if double_integ:
            y = y - v
        Y.append(y)

    Y_tensor = torch.tensor(Y)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    X_1 = torch.tensor(X).to(device)
    Y_1 = torch.tensor(Y_tensor[:,0]).view(-1, POLICY_DIM).to(device)
    X_2 = torch.tensor(X).to(device)
    Y_2 = torch.tensor(Y_tensor[:,1]).view(-1, POLICY_DIM).to(device)

    print(X_1.shape)
    print(X_2.shape)
    print(Y_1.shape)
    print(Y_2.shape)

    if not double_integ:
        torch.save(X_1, './data/x1.pt')
        torch.save(X_2, './data/x2.pt')
        torch.save(Y_1, './data/y1.pt')
        torch.save(Y_2, './data/y2.pt')
    else:
        torch.save(X_1, './data/x1_double_integ.pt')
        torch.save(X_2, './data/x2_double_integ.pt')
        torch.save(Y_1, './data/y1_double_integ.pt')
        torch.save(Y_2, './data/y2_double_integ.pt')

    return X_1, Y_1, X_2, Y_2


'''
Generate training samples for potential field policy
X: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
Y: [vx1', vy1'] and [vx2', vy2'] OR
   [ax1', ay1'] and [ax2', ay2']

Parameters:
state_space_lbs     : lower bound of state space
state_space_ubs     : upper bound of state space
total_num_agents    : total number of agents in the system
current_agent_id    : id of the agent we are generating samples for
num_samples         : number of training examples to generate
'''
def generate_potential_field_sample(
    state_space_lbs,
    state_space_ubs,
    total_num_agents,
    current_agent_id,
    num_samples = 2000000
):
    
    X = np.random.uniform(
        low=state_space_lbs, high=state_space_ubs, size=(num_samples, total_num_agents)
    )

    x_index = current_agent_id * 4
    y_index = x_index + 1

    Y = [[
        max(min(1 + 2 * X[i,x_index] / (X[i,x_index] ** 2 + X[i,y_index] ** 2), 1), -1),
        max(
        min(
            X[i,y_index] / (X[i,x_index] ** 2 + X[i,y_index] ** 2)
            + np.sign(X[i,y_index])
            * 2
            * (1 + np.exp(-(0.5 * X[i,x_index] + 2))) ** -2
            * np.exp(-(0.5 * X[i,x_index] + 2)),
            1,
        ),
        -1,
        )] for i in range(num_samples)]
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    return X, Y