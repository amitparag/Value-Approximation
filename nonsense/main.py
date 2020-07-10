import numpy as np
import crocoddyl
import torch
from tqdm import tqdm
from data import Solver
from data import Datagen
from networks import TerminalModelUnicycle
from networks import WarmstartNetwork
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
torch.set_default_dtype(torch.double)
torch.cuda.empty_cache()


class Irepa():

    """Irepa training"""
    def __init__(self, device:str = 'cpu', runs:int = 1, batch_size:int = 32, epochs:int = 2000,
            lr:float = 0.001, n_traj:int = 150, horizon:int = 100, weights:list = [1., 1.],
            precision:int = 1e-7, maxiters:int = 1000, fc1_dims:int = 20,
            fc2_dims:int = 20, fc3_dims:int = 2, weight_decay:float = 0.):

        self.device         = torch.device(device)
        self.runs           = runs
        self.batch_size     = batch_size
        self.epochs         = epochs
        self.lr             = lr
        self.n_traj         = n_traj
        self.horizon        = horizon
        self.weights        = weights
        self.precision      = precision
        self.maxiters       = maxiters
        self.fc1_dims       = fc1_dims
        self.fc2_dims       = fc2_dims
        self.fc3_dims       = fc3_dims
        self.weight_decay   = weight_decay



    def run(self):
        print(f"\n..............................Starting {self.runs} Irepa Iterations.......................................\n")
        
        # Initialize an empty net
        net = WarmstartNetwork(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, fc3_dims=self.fc3_dims, horizon=self.horizon)
        print(net)

        start = time.time()
        n_points = int(self.runs * self.n_traj)

        starting_points = Datagen.griddedData(n_points=n_points)
        np.random.shuffle(starting_points)

        start_index = 0
        end_index = self.n_traj

        for i in range(self.runs):
            print(f"\nIteration # {i+1}.")

            if i == 0:
                terminal_model_unicycle = None
            else:
                terminal_model_unicycle = TerminalModelUnicycle(net)
            
            net = self._train(starting_configs = starting_points[start_index:end_index,:],
                             neural_net=net,
                             terminal_unicycle=terminal_model_unicycle,
                             iteration = i)
            
            start_index = start_index + self.n_traj
            end_index = end_index + self.n_traj
            #torch.save(net, f'./networks/exp1/net'+str(i)+'.pth')

        end = time.time()
        print("Total Time : ", end - start)


    def _train(self,
               starting_configs,
               neural_net,
               terminal_unicycle,
               iteration= 0):

        """
        Args
        .........
                1: neural_net            : neural network to be trained
                2: starting_configs      : trajectories will be generated from these starting points.
                3: terminal_unicycle     : terminal model for irepa > 0
                4: itertion              : irepa iteration number

        Returns
        ........
                1: A trained neural net
        
        The training process:

        1: Use the given starting config to generate dataset.
        2: Reserve a portion of the dataset for validation.
        3: Initialize criteria.
        4: Initialize optimizer.
        5: Set neural net to train.
        6: In the epochs loop, send the neural net and data to device.
        7: Validate at the end of each epoch.
        8: Warmstart and compare results.

        """
        ############################################################    Training        ##################################################################################        


        # Step 1:
        x, y1, y2, y3 = Solver(starting_configs = starting_configs, 
                              terminal_model    = terminal_unicycle,
                              weights           = self.weights,
                              horizon           = self.horizon,
                              precision         = self.precision,
                              maxiters          = self.maxiters).policiesValues(as_tensor=True)

        xtest  = x[0:20, :]
        ytest1 = y1[0:20, :]
        ytest2 = y2[0:20, :]
        ytest3 = y3[0:20, :]


        xtrain  = x[20:, :]
        ytrain1 = y1[20:, :]
        ytrain2 = y2[20:, :]
        ytrain3 = y3[20:, :]


        # Convert to torch dataloader
        dataset = torch.utils.data.TensorDataset(xtrain, ytrain1, ytrain2, ytrain3)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True)

        # Step 2:
        criterion1 = torch.nn.MSELoss(reduction='sum')
        criterion2 = torch.nn.L1Loss(reduction='sum')
        criterion3 = torch.nn.MSELoss(reduction='mean')
        criterion4 = torch.nn.L1Loss(reduction='mean')

        # Step 3:
        optimizer = torch.optim.ASGD(neural_net.parameters(), lr = self.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=self.weight_decay)
        #optimizer = torch.optim.SGD(neural_net.parameters(), lr = self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        #optimizer = torch.optim.Adam(neural_net.parameters(), lr = self.lr, betas=[0.5, 0.9], weight_decay=self.weight_decay,amsgrad=True)

        neural_net.train()
        neural_net.to(self.device)

        for epoch in range(self.epochs):
            for data, target1, target2, target3 in dataloader:
                optimizer.zero_grad()

                data.to(device = self.device)
                target1.to(self.device)
                target2.to(self.device)
                target3.to(self.device)

                output1, output2, output3 = neural_net(data)
                # Value loss
                loss1 = criterion2(output1, target1)
                loss2 = criterion2(output2, target2)
                loss3 = criterion2(output2, target3)

                loss = 0.001*loss1 + loss2 + loss3
                
                loss.backward()
                optimizer.step()

            # calculate epoch loss
            val_p, xs_p, us_p = neural_net(xtest)
            v_mae = torch.mean(torch.abs(val_p - ytest1))
            xs_mae = torch.mean(torch.abs(xs_p - ytest2))
            us_mae = torch.mean(torch.abs(us_p - ytest3))

            print(f"Epoch {epoch + 1} :: value_Error = {v_mae}, xs_error = {xs_mae}, us_error = {us_mae}")
            



if __name__=='__main__':
    Irepa().run()







