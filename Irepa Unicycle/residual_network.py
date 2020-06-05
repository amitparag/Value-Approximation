import numpy as np
import torch
import torch.nn as nn

class ResidualNet(nn.Module):
    def __init__(self, 
                 input_dims:int    = 3,
                 residual_dims:int = 3,
                 fc1_dims:int      = 64,
                 fc2_dims:int      = 64,
                 activation        = nn.Tanh()
                ):
        
        super(ResidualNet, self).__init__()

        """
        Create a simple residual neural network with pytorch.
        
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
        #   x --> activation[layer1] ---> activation[layer2] ---> [layer3] ---->       [layer3] **2    #
        #                                                         residual               value         #
        ################################################################################################    
        """
        
        
        
        
        self.input_dims    = input_dims
        self.residual_dims = residual_dims
        self.fc1_dims      = fc1_dims
        self.fc2_dims      = fc2_dims
        self.activation    = activation
        
        # Structure
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.residual_dims)

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
        

    
    def residual(self, state):
        """
        Calculate the residual matrix from a given starting state.
        state --> activation[] ---> activation[] ---> residual matrix
        
        @param:
            1: state : input state, in case of unicycle, this is the starting position
            
        @returns:
            2: residual matrix
        
        
        """
        
        residual = self.activation(self.fc1(state)) 
        residual = self.activation(self.fc2(residual)) 
        residual = self.fc3(residual) 
        return residual
    
    def forward(self, state):
        residual = self.residual(state)
        
        if residual.dim() > 1:
            return 0.5 * residual.pow(2).sum(dim=1, keepdim=True)
        else:
            return 0.5 * residual.pow(2).sum().view(1, -1)
    
    def jacobian(self, state):
        """
        @param:
            1: state
            
        @returns
            The jacobian of the Value function with respect to state.
            Jacobian = dV/dx
        
        """
        j = torch.autograd.functional.jacobian(self.forward, state).detach().squeeze()
        return j
    
    def hessian(self, state):
        """
        @param:
            1: x = state
            
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
        h = [torch.autograd.functional.hessian(self.forward, state).detach().squeeze() for state in states]
        return torch.stack(h).squeeze()
    
    
    # Gauss Approximation .........................
    def gradient(self, state):
        """
        Gauss Approximation of the gradient:
            
            Gradient = J.T @ residual           where J = jacobian of the residual
        
        @params:
            1: state : the input to the neural network
            
        @returns
            1: gradient
        """
        # jacobian of the residual
        j = torch.autograd.functional.jacobian(self.residual, state).detach().squeeze()
        
        # residual
        r = self.residual(state)
       
        return j.T @ r.T
    
    def newton_hessian(self, state):
        """
        Gauss Approximation of the Hessian:
            Hessian = J.T @ J,         where J = jacobian of the residual
        
        """
        # jacobian of the residual    
        j = torch.autograd.functional.jacobian(self.residual, state).detach().squeeze()
    
        return j.T @ j
    
    def batch_gradient(self, states):
        grads = [self.gradient(state) for state in states]
        
        return torch.stack(grads).squeeze()
    
    def batch_newton_hessian(self, states):
        hessians = [self.newton_hessian(state) for state in states]
        
        return torch.stack(hessians).squeeze()
    
