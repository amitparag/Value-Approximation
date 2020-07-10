""" Feedforward networks to learn value function and policy"""

import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.double)

class ValueNetwork(nn.Module):

        def __init__(self, 
                     input_dims:int  = 3,
                     output_dims:int = 1,
                     fc1_dims:int    = 100,
                     fc2_dims:int    = 100,
                     fc3_dims:int    = 1,
                     activation      = nn.ReLU(),
                     device:str      = 'cpu'
                     ):


                super(ValueNetwork, self).__init__()
                """Instantiate an untrained neural network with the given params

                Args
                ........
                        
                        1: input_dims   = dimensions of the state space of the robot. 3 for unicycle
                        2: output_dims  = dimensions of the ddp.cost. 1
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
                return torch.autograd.functional.jacobian(self.forward, state).cpu().detach().numpy().reshape(1, 3)

        def hessian(self, state):
                """
                Args
                ......
                        1: x = state

                Returns
                .......
                
                        The true hessian of the Value function with respect to state, if the activation function is Tanh()
                        else Gauss-Newton Hessian        
                        Gauss-Newton Hessian = J.T @ J, this approximation works only when y_pred - y_true ~ = 0
                """
                if not isinstance(self.activation , torch.nn.modules.activation.Tanh):
                        #print("Activation :", self.activation,"Using Approximation")
                        j = self.jacobian(state)
                        h = j.T @ j
                        
                        return h
                else:
                     return torch.autograd.functional.hessian(self.forward, state).cpu().detach().squeeze().numpy().reshape(3, 3)


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

        def guess_xs(self, state, horizon=30):
                """
                Given a starting state, predict the state trajectory for the entire length of the horizon. The predicted trajectory should be of length horion +1
                """
                xs = []
                xs.append(state)

                for _ in range(horizon):
                        state = self(state)
                        xs.append(state)

                return torch.stack(xs).cpu().detach().numpy().reshape(horizon+1,3)


class Datagen:
    def griddedData(n_points:int = 2550,
                    xy_limits:list = [-1.9,1.9],
                    theta_limits:list = [-np.pi/2, np.pi/2]
                    ):
        """Generate datapoints from a grid"""

        size = int(np.sqrt(n_points)) + 1

        min_x, max_x = [*xy_limits]
        xrange = np.linspace(min_x,max_x,size, endpoint=True)
        trange = np.linspace(*theta_limits, size, endpoint=True)
        points = np.array([ [x1,x2, x3] for x1 in xrange for x2 in xrange for x3 in trange])

        np.round_(points, decimals=6)
        np.random.shuffle(points)
        points = points[0:n_points, : ]
        return points



