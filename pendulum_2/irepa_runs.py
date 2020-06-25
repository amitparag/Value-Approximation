import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from neural_net import feedForwardNet
from pendulum_utils import terminalPendulum, Solver, Datagen
import example_robot_data
torch.set_default_tensor_type(torch.DoubleTensor)

class Irepa():

    def __init__(self,
                runs       : int   = 20,
                fc1_dims   : int   = 12,
                fc2_dims   : int   = 12,
                fc3_dims   : int   = 12,
                ntraj      : int   = 100,
                lr         : float = 1e-3,
                batch_size : int   = 128,
                epochs     : int   = 1000,
                horizon    : int   = 100,
                use_fddp   : bool  = True,
                precision  : float = 1e-7,
                maxiters   : int   = 1000,
                full_traj  : bool  = True,
                th1_lims   : list  = [-np.pi/2, np.pi/2],
                th2_lims   : list  = [-np.pi/2, np.pi/2],
                vel1_lims  : list  = [-0.25, 0.25],
                vel2_lims  : list  = [-0.25, 0.25]
               ):
        
        self.runs         = runs
        self.fc1_dims     = fc1_dims
        self.fc2_dims     = fc2_dims
        self.fc3_dims     = fc3_dims
        self.ntraj        = ntraj
        self.lr           = lr
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.horizon      = horizon
        self.use_fddp     = use_fddp
        self.precision    = precision
        self.maxiters     = maxiters
        self.full_traj    = full_traj
        self.th1_lims     = th1_lims
        self.th2_lims     = th2_lims
        self.vel1_lims    = vel1_lims
        self.vel2_lims    = vel2_lims


    def _training_data(self, terminal_model = None):
        
        # Generate training data for first irepa run, i.e training on data not generated with terminal pendulum
        random_starting_configs = Datagen.samples(size = self.ntraj,
                                                  sampling='uniform',
                                                 th1_lims = self.th1_lims,
                                                 th2_lims = self.th2_lims,
                                                 vel1_lims = self.vel1_lims,
                                                 vel2_lims = self.vel2_lims)
        
        xtrain, ytrain = Datagen.statesValues(init_states   = random_starting_configs,
                                             terminal_model = terminal_model,
                                             horizon        = self.horizon,
                                             use_fddp       = self.use_fddp,
                                             full_traj      = self.full_traj,
                                             as_tensor      = True)
        
        
        
        
        return xtrain, ytrain


    def _training(self, net, xtrain, ytrain):
        """
        @params:
            1: net = neural net to be trained
            2: xtrain, ytrain = dataset

        @returns:
            1: trained neural network

        """
        # Convert to torch dataloader
        dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size)

        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(net.parameters(), lr = self.lr)  

        net.train()

        for epoch in tqdm(range(self.epochs)):        
            for data, target in dataloader: 

                output = net(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        del dataset, dataloader, xtrain, ytrain
        return net

    def training(self):
        print(f"Starting {self.runs} Irepa runs for Feedforward Network.......\n")

        # Instantiate a neural net
        net = feedForwardNet(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, fc3_dims=self.fc3_dims)
        
        # irepa1:
        xtrain, ytrain = self._training_data(terminal_model=None)
        print(f"Starting 0th run .......\n")

        net = self._training(net, xtrain, ytrain)
        torch.save(net, "./nets/net1.pth")
        del xtrain, ytrain

        for i in tqdm(range(self.runs - 1)):
            # Generate training data with neural network inside crocoddyl
            print(f" Run {i + 1} .......\n")

            robot = example_robot_data.loadDoublePendulum()
            robot_model = robot.model
            terminal_model = terminalPendulum(net, robot_model)
            
            xtrain, ytrain = self._training_data(terminal_model = terminal_model)
            
            net = self._training(net = net,
                                 xtrain=xtrain,
                                 ytrain=ytrain)

            
            torch.save(net, './nets/net'+str(i+2)+'.pth')
            
            del terminal_model, xtrain, ytrain
            
        print("Done........")                      
        
if __name__=='__main__':
    

    irepa = Irepa()

    irepa.training()
