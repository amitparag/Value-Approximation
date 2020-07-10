"""A generic feedforward net to rollout policy from an initial starting point."""

import torch
import numpy as np
import crocoddyl
import torch.nn as nn
torch.set_default_dtype(torch.double)


class PolicyNetwork(nn.Module):
    def __init__(self,
        state_dims:int  = 3,
        horizon:int     = 50,
        policy_dims:int = 3,
        fc1_dims:int    = 20,
        fc2_dims:int    = 20,
        fc3_dims:int    = 3,
        activation      = nn.ReLU(),
        device:str      = 'cpu'
        ):
        super(PolicyNetwork, self).__init__()

        self.device         = torch.device(device)
        self.state_dims     = state_dims
        self.rollout_dims   = horizon * policy_dims
        self.horizon        = horizon
        self.policy_dims    = policy_dims
        self.fc1_dims       = fc1_dims
        self.fc2_dims       = fc2_dims
        self.fc3_dims       = fc3_dims
        self.activation     = activation
        self.device         = device

        # ...... Initialize layers
        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.rollout_dims)


        #....... Weight Initialization protocol
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        #....... Bias Initialization protocol
        nn.init.constant_(self.fc1.bias, 0.003)
        nn.init.constant_(self.fc2.bias, 0.003)
        nn.init.constant_(self.fc3.bias, 0.003)

        self.to(device)


    def forward(self, state):
        policy = self.activation(self.fc1(state))
        policy = self.activation(self.fc2(policy))
        policy = self.activation(self.fc3(policy))
        policy = self.fc4(policy)
        return policy

    def guessAPolicy(self, state):
        """
        A helper function to reshape policy and return the guess as np.array
        """
        
        policy = self.forward(state).cpu().detach().numpy().reshape(self.horizon, self.policy_dims)
        policy[0] = state.cpu().detach().numpy()
        return policy


