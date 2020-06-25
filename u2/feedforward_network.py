
"""
A feedforward neural network to predict value functions.

"""
import numpy as np
import torch 
import torch.nn as nn

class FeedForwardNet(nn.Module):
    def __init__(self, 
                 input_dims:int = 3,
                 out_dims:int   = 1,
                 fc1_dims:int   = 20,
                 fc2_dims:int   = 20,
                 activation     = nn.Tanh()
                ):
        super(FeedForwardNet, self).__init__()
        
        """
        Create a simple feedforward neural network with pytorch.
        
        @params:
            1: input_dims  = input_features, i.e the number of features in the training dataset
            2: fc1_dims    = number of units in the first fully connected layer. Default 64
            3: fc2_dims    = number of units in the second fully connected layer. Default 64
            4: activation  = activation for the layers, default tanh.
            
        @returns:
            A 3 layered neural network
            
            
        ################################################################################################
        #   The architecture of the network is :                                                       #
        #                                                                                              #
        #   x --> activation[layer1] ---> activation[layer2] ---> [layer3] == Value                    #
        #                                                                                              #
        ################################################################################################
            
        """
        
        self.input_dims = input_dims
        self.out_dims   = out_dims
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.activation = activation
        
        # Structure
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.out_dims)

        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        
        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        
        # Send the neural net to device
        self.device = torch.device('cpu')
        self.to(self.device)
        
    def forward(self, state):
        """
        The Value function predicted by the neural net. 
        
        """
        value = self.activation(self.fc1(state))
        value = self.activation(self.fc2(value))
        value = self.fc3(value)
        
        return value
    
    def jacobian(self, state):
        """
        @param:
            x = state
            
        @return
            The jacobian of the Value function with respect to state.
            Jacobian = dV/dx
        
        """
        return torch.autograd.functional.jacobian(self.forward, state).detach().squeeze()
    
    def hessian(self, state):
        """
        @param:
            x = state
            
        @return
            The hessian of the Value function with respect to state.
            Hessian = d^2V/dx^2        
        """
        return torch.autograd.functional.hessian(self.forward, state).detach().squeeze()
        

    def batch_jacobian(self, states):
        """
        Returns the jacobians of multiple inputs
        """
        j = [torch.autograd.functional.jacobian(self.forward, state).detach().squeeze() for state in states]
        return torch.stack(j).squeeze()
    
    def batch_hessian(self, states):
        """
        Returns the hessians of the multiple inputs 
        
        """
        h = [torch.autograd.functional.hessian(self.forward, states).detach().squeeze() for states in states]
        return torch.stack(h).squeeze()
    
