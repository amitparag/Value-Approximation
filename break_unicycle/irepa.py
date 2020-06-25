"""
Implementation of Irepa iterations using crocoddyl.

Some terminology:

1: Nominal_crocoddyl: Regular crocoddyl with ActionModelUnicycle running with loss function, as terminal model.
                      This is equivalent to:
                          model = crocoddyl.ActionModelUnicycle()
                          problem = crocoddyl.ShootingProblem(x0.T, [model]*T, model)
                          
2: Terminal_crocoddyl: Crocoddyl after a large number of iterations. This is the model 
                       used to establish convergence. It will correspond to the last irepa run.
                       
3: Running_crocoddyl : Crocodddyl in its ith iteration. If we run irepa for 50 iterations, then all iterations
                       before terminal crocoddyl are considered to be running iterations.

The algorithm is as follows:

    1: Generate dataset from Nominal_crocoddyl. 
    2: Train the neural network on the dataset.
    
    3: Use neural net inside crocoddyl to generate new dataset. This is the 1st Running_crocoddyl
    4: Train the neural net.
    
    Repeat 3, 4 until convergence to Terminal_crocoddyl

Three cases are considered for convergence.

1: similarity to 0              -------------> Similarity of any Running_crocoddyl to Nominal_crocoddyl
2: similarity to asymptote      -------------> Similarity of any Running_crocoddyl to Terminal_crocoddyl
3: similarity to previous iteration  --------> Similarity of nth Running_crocoddyl to (n-1)th Running_crocoddyl

The (dis)similarity measures can be established in two ways.

1: Mean Squared errors 
2: Procustes dissmilarity

"""


import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_generator import Datagen
from terminal_unicycle import FeedforwardUnicycle, ResidualUnicycle, zeroCostUnicycle
from feedforward_network import FeedForwardNet
from residual_network import ResidualNet

class Irepa:
    """
    Training for irepa iterations
    """
    def _feedforward_net(fc1_dims = 64,
                         fc2_dims = 64):
        
        
        """
        Instantiate and return a feedforward neural network with the given dimensions.
        The activation used is nn.Tanh(), since the network needs to be double differentiable
        
        @params:
            1: fc1_dims      = number of hidden units in the first fully connected layer
            2: fc2_dims      = number of hidden units in the second fully connected layer.
            
        @returns:
            A fully connected feed forward layer
        """
        fnet = FeedForwardNet(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        return fnet
    
   
    def feed_forward_training(fc1_dims=64,
                              fc2_dims=64,
                              runs = 50,
                              ntraj= 100,
                              lr=1e-3, 
                              batch_size=128,
                              epochs=1000,
                              ):
                
        """
        The main irepa training loop
        @params:
            1: fc1_dims   = Hidden units in layer 1 of feedforward net
            2: fc2_dims   = Hidden units of layer 2
            2: runs       = 50. Number of trainings
            3: ntraj      = 100, number of trajectories used for training
            4: lr         = learning rate
            5: batch_size = 128
            6: epochs     = 1000
            
        @returns:
           trained feedforward neural network
        
        """
        print(f"Starting {runs} Irepa runs for Feedforward Network.......\n")
        zero_cost = zeroCostUnicycle()
        # Get training data from nominal crocoddyl
        xtrain, ytrain = Datagen.data(ntrajectories=ntraj, terminal_model=zero_cost)
        
        # Instantiate a feedforward network with the given fc dims
        net = Irepa._feedforward_net(fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        
        # Irepa 0
        net = Irepa._training(net = net,
                              xtrain=xtrain,
                              ytrain=ytrain,
                              lr=lr,
                              batch_size=batch_size,
                              epochs=epochs)
        
        torch.save(net, "./Fnet/net1.pth")
        del xtrain, ytrain
        # main loop
        for i in range(runs-1):
            
            # Generate training data with neural network inside crocoddyl
            terminal_model = FeedforwardUnicycle(net)
            xtrain, ytrain = Datagen.data(terminal_model = terminal_model,
                                          ntrajectories = ntraj)
            
            net = Irepa._training(net = net,
                              xtrain=xtrain,
                              ytrain=ytrain,
                              lr=lr,
                              batch_size=batch_size,
                              epochs=epochs)
            
            torch.save(net, './Fnet/net'+str(i+2)+'.pth')
            
            del terminal_model, xtrain, ytrain
            
        print("Done........")
        
           

        
    def _training( net, xtrain, ytrain,lr, batch_size, epochs):
        """
        @params:
            1: net = neural net to be trained
            2: xtrain, ytrain = dataset
            
        @returns:
            1: trained neural network
            
        """
        # Convert to torch dataloader
        dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
        
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(net.parameters(), lr = lr)  
        
        net.float()
        net.train()
        
        for epoch in tqdm(range(epochs)):        
            for data, target in dataloader: 

                output = net(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        del dataset, dataloader, xtrain, ytrain
        return net


if __name__=='__main__':
    Irepa.feed_forward_training(epochs=1000, runs=20, ntraj=100)

