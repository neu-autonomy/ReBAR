import torch.nn as nn

STATE_DIM = 2
POLICY_DIM = 2

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

'''
defines the original NNC structure
The NNCs are trained in decentralized fashion
'''
def two_agent_orig(HIDDEN1_DIM=10, HIDDEN2_DIM=10, n=2):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM*n+POLICY_DIM*n, HIDDEN1_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM, HIDDEN2_DIM),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM, POLICY_DIM)
    )
    return model


'''
defines the NNC with control signal projection onto [u_lb, u_ub]
'''
def two_agent_nonres_ulimits(HIDDEN1_DIM=10, HIDDEN2_DIM=10, n=2):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM*n+POLICY_DIM*n, HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n),
        nn.ReLU(),
        nn.Linear(HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n, HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n),
        nn.ReLU(),
        nn.Linear(HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n, POLICY_DIM + STATE_DIM*n+POLICY_DIM*n),
        nn.ReLU(),
        nn.Linear(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n, POLICY_DIM + STATE_DIM*n+POLICY_DIM*n),
        nn.ReLU(),
        nn.Linear(POLICY_DIM + STATE_DIM*n+POLICY_DIM*n, STATE_DIM + POLICY_DIM)
    )
    return model


'''
defines the larger ReLU MLP representing the merged NNCs
'''
def two_agent_nonres_ulimits_merged(HIDDEN1_DIM=10, HIDDEN2_DIM=10, n=2):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(STATE_DIM*n+POLICY_DIM*n, 2 * (HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n)),
        nn.ReLU(),
        nn.Linear(2 * (HIDDEN1_DIM + STATE_DIM*n+POLICY_DIM*n), 2 * (HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n)),
        nn.ReLU(),
        nn.Linear(2 * (HIDDEN2_DIM + STATE_DIM*n+POLICY_DIM*n), 2 * (POLICY_DIM + STATE_DIM*n+POLICY_DIM*n)),
        nn.ReLU(),
        nn.Linear(2 * (POLICY_DIM + STATE_DIM*n+POLICY_DIM*n), 2 * (POLICY_DIM + STATE_DIM*n+POLICY_DIM*n)),
        nn.ReLU(),
        nn.Linear(2 * (POLICY_DIM + STATE_DIM*n+POLICY_DIM*n), 2 * (STATE_DIM + POLICY_DIM))
    )
    return model