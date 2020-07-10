import numpy as np
import torch
import torch.nn as nn
torch.set_default_dtype(torch.double)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

class MarkovNetwork(nn.Module):

        def __init__(self, 
                     input_dims:int  = 3,
                     output_dims:int = 3,
                     fc1_dims:int    = 100,
                     fc2_dims:int    = 100,
                     fc3_dims:int    = 1,
                     activation      = nn.ReLU6(),
                     device:str      = 'cpu'
                     ):


                super(MarkovNetwork, self).__init__()
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
                The trajectory predicted by the neural net. 
                
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

                trajectory =  torch.stack(xs).cpu().detach().numpy().reshape(horizon+1,3)
                print(trajectory.shape)

class PolicyNetwork(nn.Module):
        def __init__(self, 
                     input_dims:int  = 3,
                     policy_dims:int = 90,   
                     fc1_dims:int    = 100,
                     fc2_dims:int    = 100,
                     fc3_dims:int    = 2,
                     activation      = nn.ReLU6(),
                     device:str      = 'cpu'
                     ):


                super(PolicyNetwork, self).__init__()
                """Instantiate an untrained neural network with the given params

                Args
                ........
                        
                        1: input_dims   = dimensions of the state space of the robot. 3 for unicycle
                        2: policy_dims  = maximum dimensions of the trajectory
                        3: fc1_dims     = number of units in the first fully connected layer. Default 100
                        4: fc2_dims     = number of units in the second fully connected layer. Default 100
                        5: fc3_dims     = number of units in the third fully connected layer. Default 1
                        6: activation   = activation for the layers, default ReLU.
                        7: device       = device for computations. Generally CPU

                """

                self.input_dims    = input_dims
                self.policy_dims   = policy_dims
                self.fc1_dims      = fc1_dims
                self.fc2_dims      = fc2_dims
                self.fc3_dims      = fc3_dims
                self.activation    = activation


                #........... Structure
                self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
                self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
                self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
                self.fc4 = nn.Linear(self.fc3_dims, self.policy_dims)

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

        def forward(self, states):
                """
                The trajectory predicted by the neural net. 
                
                """
                if states.dim() != 2:
                        states = states.reshape(1, -1)
                for state in states:

                        #print(state.dim())
                        leng_traj = int(state[-1])
                        #print(leng_traj)
                        start_state = state[0:-1]
                        #print(start_state)   
                        trajectory = self.activation(self.fc1(start_state))
                        trajectory = self.activation(self.fc2(trajectory))
                        trajectory = self.activation(self.fc3(trajectory))
                        trajectory = self.fc4(trajectory)
                        
                        return trajectory[0:leng_traj]




