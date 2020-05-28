import numpy as np
import torch
import torch.nn as nn

class ResidualNet(nn.Module):
    def __init__(self, 
                 in_features:int  = 3,
                 out_features:int = 3,
                 n_hiddenUnits:int = 128,
                 activation = nn.Tanh()):
        
        """
        Create a simple residual neural network with pytorch.
        
        @params:
            1: in_features  = input_features, i.e the number of features in the training dataset
            2: out_features = output_features, i.e the size of the residual layer. Default to 3
            3: nhiddenunits = number of units in a hidden layer. Default 128
            4: activation   = activation for the layers, default tanh.
            
        @returns:
            A 3 layered residual neural network.
            
            
        ################################################################################################
        #   The architecture of the network is :                                                       #
        #                                                                                              #
        #   x --> activation[layer1] ---> activation[layer2] ---> [layer3] ---->       [layer3] **2    #
        #                                                         residual               value         #
        ################################################################################################    
        """
        
        
        super(ResidualNet, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_hiddenUnits = n_hiddenUnits
        
        # Structure
        self.fc1 = nn.Linear(self.in_features, self.n_hiddenUnits)
        self.fc2 = nn.Linear(self.n_hiddenUnits, self.n_hiddenUnits)
        self.fc3 = nn.Linear(self.n_hiddenUnits, self.out_features)

        # Weight Initialization protocol
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        
        # Bias Initialization protocol
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)
        
        # Activation
        self.activation = activation
      
        self.device = torch.device('cpu')
        self.to(self.device)

        
        
    def forward(self, x):
        """
        output = sum (residual(x) ** 2)
        """
        x = self.residual(x)
        
        if x.dim() > 1:
            return 0.5 * x.pow(2).sum(dim=1, keepdim=True)
        else:
            return 0.5 * x.pow(2).sum().view(1,-1)
    
    def residual(self, x):
        """
        x --> activation[] ---> activation[] ---> residual matrix
        
        """
        
        x = self.activation(self.fc1(x)) 
        x = self.activation(self.fc2(x)) 
        x = self.fc3(x) 
        return x
        
    def jacobian(self, x):
        """
        Returns the jacobian of the value , i.e jacobian of neural_net.forward(x), w.r.t x.
        This is the true jacobian of the neural network.
        Should be used only when x is a single tensor
        
        @params:
            1. x = input
        
        @returns
            1. d(V)/d(x)
        
        """
        j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
        return j
    
    
    def batch_jacobian(self, x):
        """
        Wrapper around self.jacobian_value for multiple inputs
        
        @params:
            1; x  = input array
        @returns:
            1: tensor array of jacobians
        """
        j = []
        for xyz in x:
            j.append(self.jacobian(xyz))
        return torch.stack(j).squeeze()
    
    def hessian(self, x):
        """
        Returns the Hessian of the value , i.e jacobian of neural_net.forward(x), w.r.t x.
        This is the true hessian of the neural network.
        Should be used only when x is a single tensor
        
        @params:
            1. x = input
        
        @returns
            1. d2(V)/d(x2)
        
        """
        h = torch.autograd.functional.hessian(self.forward, x).squeeze()
        return h
    
    def batch_hessian(self, x):
        """
        Wrapper around self.hessian_value for multiple inputs
        
        @params:
            1; x  = input array
        @returns:
            1: 3-d tensor array of hessians
        """
        h = []
        for xyz in x:
            h.append(self.hessian(xyz))
        return torch.stack(h).squeeze()
    
    #.....Gauss Approximation
    
    def jacobian_residual(self, x):
        """
        Returns the jacobian of the residual
        """
        j = torch.autograd.functional.jacobian(self.residual, x).squeeze()
        return j
    
    def gradient(self, x):
        """
        Gauss Approximation of the gradient:
            Gradient = J.T @ residual
                where J = jacobian of the residual
        
        """
        j = self.jacobian_residual(x)
        r = self.residual(x)
        return j.T @ r
    
    def approx_hessian(self, x):
        """
        Gauss Approximation of the Hessian:
            Hessian = J.T @ J
                where J = jacobian of the residual
        
        """
        j = self.jacobian_residual(x)
        return j.T @ j
   
    def batch_gradient(self, x):
        """
        Calculates the batch gradient for a given batch
        
        """
        
        grad = []
        for x in x:
            
            j = self.gradient(x)
            grad.append(j)
        return torch.stack(grad).squeeze()
    
    
    def batch_approx_hessian(self, x):
        """
        Calculates the batch hessian for a given batch
        
        """
        
        grad = []
        for x in x:
            
            j = self.approx_hessian(x)
            grad.append(j)
        return torch.stack(grad).squeeze()
        