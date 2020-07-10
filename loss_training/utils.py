"""
A feedforward neural network to approximate value functions.
"""
import numpy as np
import torch 
import torch.nn as nn
import crocoddyl
torch.set_default_dtype(torch.double)


class ValueNetwork(nn.Module):
    def __init__(self, 
                 input_dims:int = 3,
                 out_dims:int   = 1,
                 fc1_dims:int   = 20,
                 fc2_dims:int   = 26,
                 fc3_dims:int   = 2,
                 activation     = nn.Tanh(),
                 device         = 'cpu'
                ):
        super(ValueNetwork, self).__init__()
        
        """
        Create a simple feedforward neural network with pytorch.
        
        Args
        ......
                1: input_dims  = input_features, i.e the number of features in the training dataset
                2: fc1_dims    = number of units in the first fully connected layer. Default 20
                3: fc2_dims    = number of units in the second fully connected layer. Default 20
                4: fc3_dims    = number of units in the third fully connected layer. Default 20
                5: activation  = activation for the layers, default tanh.
            
        Return
        ......
                1: A fully connected 3 hidden layered neural network
            
            
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
        self.fc3_dims   = fc3_dims
        self.activation = activation
        
        #........... Structure
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.out_dims)

        #........... Weight Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        
        #........... Bias Initialization protocol
        nn.init.constant_(self.fc1.bias, 0.001)
        nn.init.constant_(self.fc2.bias, 0.001)
        nn.init.constant_(self.fc3.bias, 0.001)
        nn.init.constant_(self.fc4.bias, 0.001)
        
        # Send the neural net to device
        self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, state):
        """
        The Value function predicted by the neural net. 
        
        """
        value = self.activation(self.fc1(state))
        value = self.activation(self.fc2(value))
        value = self.activation(self.fc3(value))
        value = self.fc4(value)
        
        return value
    
    def jacobian(self, state):
        """
        Args
        ......
                1: x = state
            
        Returns
        .......
                1: The jacobian of the Value function with respect to state. Jacobian = dV/dx
        
        """
        return torch.autograd.functional.jacobian(self.forward, state).detach().squeeze()
    
    def hessian(self, state):
        """
        Args
        ......
                1: x = state

        Returns
        .......
            
                1: The hessian of the Value function with respect to state. Hessian = d^2V/dx^2        
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
            data.cost = self.net(x).item()
    
    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone

        x = torch.Tensor(x).resize_(1, 3)
        
        data.Lx = self.net.jacobian(x).detach().numpy()
        data.Lxx = self.net.hessian(x).detach().numpy()
        
        
class PolicyNetwork(nn.Module):

        def __init__(self, 
                     input_dims:int  = 3,
                     output_dims:int = 3,
                     fc1_dims:int    = 100,
                     fc2_dims:int    = 100,
                     fc3_dims:int    = 1,
                     activation      = nn.ReLU6(),
                     device:str      = 'cpu'
                     ):


                super(PolicyNetwork, self).__init__()
                """Instantiate an untrained neural network with the given params

                Args
                ........
                        
                        1: input_dims   = dimensions of the state space of the robot. 3 for unicycle
                        2: output_dims  = dimensions of the next state
                        3: fc1_dims     = number of units in the first fully connected layer. Default 100
                        4: fc2_dims     = number of units in the second fully connected layer. Default 100
                        5: fc3_dims     = number of units in the third fully connected layer. Default 1
                        6: activation   = activation for the layers, default ReLU.
                        7: device       = device for computations. Generally CPU

                """

                self.input_dims    = input_dims
                self.output_dims   = output_dims
                self.fc1_dims      = fc1_dims
                self.fc2_dims      = fc2_dims
                self.fc3_dims      = fc3_dims
                self.activation    = activation


                #........... Structure
                self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
                self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
                self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
                self.fc4 = nn.Linear(self.fc3_dims, self.output_dims)

                #........... Weight Initialization protocol
                nn.init.xavier_uniform_(self.fc1.weight)
                nn.init.xavier_uniform_(self.fc2.weight)
                nn.init.xavier_uniform_(self.fc3.weight)
                nn.init.xavier_uniform_(self.fc4.weight)

                
                #........... Bias Initialization protocol
                nn.init.constant_(self.fc1.bias, 0.003)
                nn.init.constant_(self.fc2.bias, 0.003)
                nn.init.constant_(self.fc3.bias, 0.003)
                nn.init.constant_(self.fc4.bias, 0.003)
                
                # Send the neural net to device
                self.device = torch.device(device)
                self.to(self.device)


                
        def forward(self, state):
                """
                The Value function predicted by the neural net. 
                
                """
                next_state = self.activation(self.fc1(state))
                next_state = self.activation(self.fc2(next_state))
                next_state = self.activation(self.fc3(next_state))
                next_state = self.fc4(next_state)
                
                return next_state

        def guess_xs(self, state, horizon=20):
                """
                Given a starting state, predict the state trajectory for the entire length of the horizon. The predicted trajectory should be of length horion +1
                """
                xs = []
                xs.append(state)

                for _ in range(horizon):
                        state = self(state)
                        xs.append(state)

                return torch.stack(xs).cpu().detach().numpy().reshape(horizon+1,3)
