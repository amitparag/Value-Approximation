"""
A simple multioutput feedforward neural network to predict value function, trajectory and control given an initial starting state.

The data must be of the following format:

xtrain = states_tensor of size (ntrajectories, 3)
ytrain1 = values_tensor of size (ntrajectories, 1)
ytrain2 = xs_tensor of size (ntrajectories, horizon)
ytrain3 = us_tensor of size (ntrajectories, horizon)

To use just the value net, either don't use ytrain2 and ytrain3 in the loss function optimization or set horizon to 0, which should give a pure ValueNet.
"""
import torch
import numpy as np
import crocoddyl
import torch.nn as nn
torch.set_default_dtype(torch.double)

class WarmstartNetwork(nn.Module):
    def __init__(self,
                state_dims:int  = 3,
                value_dims:int  = 1,
                horizon:int     = 30,
                fc1_dims:int    = 20,
                fc2_dims:int    = 20,
                fc3_dims:int    = 3,
                activation      = nn.Tanh(),
                ):
        super(WarmstartNetwork, self).__init__()


        """
        Create a simple feedforward neural network with pytorch.
        
        @params:
            1: state_dims  = input_features, i.e the number of features in the training dataset. This correponds to the dimension of the state space for the problem
            2: value_dims  = corresponds to ddp.cost. features. should be 1 for value
            3; horizon     = corresponds to ddp.xs, used to establish output shape when xs is being learned
            4: fc1_dims    = number of units in the first fully connected layer. Default 20
            5: fc2_dims    = number of units in the second fully connected layer. Default 20
            6: fc3_dims    = number of units in the second fully connected layer. Default 20
            7: activation  = activation for the layers, default tanh.
            
        @returns:
            A 3 layered neural network with three outputs
            
            
        #########################################################################################################
        #   The architecture of the network is :                                                                #
        #                                                                                                       #
        #   x --> activation[layer1] ---> activation[layer2] ---> [layer3] == Value, Trajectory, Control        #
        #                                                                                                       #
        #########################################################################################################

        To use this as a simple value net, set horizon = 0. 


        """

        self.state_dims = state_dims
        self.value_dims = value_dims
        self.xs_dims    = horizon*3
        self.us_dims    = horizon*3
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.fc3_dims   = fc3_dims
        self.activation = activation


        self.fc1 = nn.Linear(self.state_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        
        # Value Output Layer
        self.fc4 = nn.Linear(self.fc3_dims, self.value_dims)
        # Trajectory Output Layer
        self.fc5 = nn.Linear(self.fc3_dims, self.xs_dims)
        # Control Output Layer
        self.fc6 = nn.Linear(self.fc3_dims, self.us_dims)

        # Weight Initialization protocol
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)


        
        # Bias Initialization protocol
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fc4.bias, 0)
        nn.init.constant_(self.fc5.bias, 0)
        nn.init.constant_(self.fc6.bias, 0)



    def forward(self, state):
        """
        Calculate the value, trajectory and control from this state.
        output[0] = value
        output[1] = xs
        output[2] = us

        To view the trajectory if state is a single vector (not in batches), use: view(-1, 3)
        else, for i in output[1]: i.view(-1, 3) should do the trick.
       
        """

        output = self.activation(self.fc1(state))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        
        value = self.fc4(output)
        trajectory = self.fc5(output)
        control = self.fc5(output)

        if self.xs_dims == 0: return value
        else: return value, trajectory, control


    def value_function(self, state):
        """Return just the value function"""
        output = self.activation(self.fc1(state))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        value = self.fc4(output)

        return value

    def jacobian(self, state):
        """
        @Args:
            x = state
            
        @Returns
            The jacobian of the Value function with respect to state.
            Jacobian = dV/dx
        
        """
        return torch.autograd.functional.jacobian(self.value_function, state).squeeze()


    def hessian(self, state):
        """
        @Args:
            x = state
            
        @Returns
            The hessian of the Value function with respect to state.
            Hessian = d^2V/dx^2        
        """
        return torch.autograd.functional.hessian(self.value_function, state)

    def batch_jacobian(self, states):
        """
        Returns the jacobians of multiple inputs
        """
        j = [torch.autograd.functional.jacobian(self.value_function, state).squeeze() for state in states]
        return torch.stack(j).squeeze()
    
    def batch_hessian(self, states):
        """
        Returns the hessians of the multiple inputs 
        
        """
        h = [torch.autograd.functional.hessian(self.value_function, state).squeeze() for state in states]
        return torch.stack(h).squeeze()        

class TerminalModelUnicycle(crocoddyl.ActionModelAbstract):
    """
    This includes a feedforward network in crocoddyl
    
    """
    def __init__(self, neural_net):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = neural_net
        device = torch.device('cpu')
        self.net.to(device)

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        x = torch.Tensor(x).resize_(1, 3)
        
        with torch.no_grad():
            data.cost = self.net.value_function(x).item()
    
    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone

        x = torch.Tensor(x).resize_(1, 3)
        
        data.Lx = self.net.jacobian(x).detach().numpy()
        data.Lxx = self.net.hessian(x).detach().numpy()

x = torch.rand(10, 3)
net = WarmstartNetwork()
net(x)