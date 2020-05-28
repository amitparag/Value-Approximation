import torch 
import numpy as np
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, 
                 in_features:int  = 3,
                 out_features:int = 1,
                 n_hiddenUnits:int = 64,
                 activation = nn.Tanh()
                ):
        
        
        """
        Create a simple feedforward neural network with pytorch.
        
        @params:
            1: in_features  = input_features, i.e the number of features in the training dataset
            2: out_features = output_features, i.e the number of output features in the training dataset
            3: nhiddenunits = number of units in a hidden layer. Default 64
            4: activation   = activation for the layers, default tanh.
            
        @returns:
            A 3 layered neural network
            
            
        ################################################################################################
        #   The architecture of the network is :                                                       #
        #                                                                                              #
        #   x --> activation[layer1] ---> activation[layer2] ---> [layer3] == Value                    #
        #                                                                                              #
        ################################################################################################
            
        """
        
        super(ValueNet, self).__init__()
        
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
        x --> activation[] ---> activation[] ---> output
        
        """
        
        x = self.activation(self.fc1(x)) 
        x = self.activation(self.fc2(x)) 
        x = self.fc3(x) 
        
        return x
    

    def jacobian(self, x):
        """
        Calculate and return the jacobian of neural network output with respect to a single input
        
        """
        j = torch.autograd.functional.jacobian(self.forward, x).squeeze()
        return j
    
    
    def batch_jacobian(self, x):
        """
        Returns the jacobians of multiple inputs
        """
        j = [torch.autograd.functional.jacobian(self.forward, x) for x in x]
        return torch.stack(j).squeeze()
    
    

    def hessian(self, x):
        """
        Calculate and return the hessian of the neural network prediction with respect to a single input
        
        """
        h = torch.autograd.functional.hessian(self.forward, x).squeeze()
        return h

    def batch_hessian(self, x):
        """
        Returns the hessians of the multiple inputs 
        
        """
        h = [torch.autograd.functional.hessian(self.forward, x) for x in x]
        return torch.stack(h).squeeze()
    