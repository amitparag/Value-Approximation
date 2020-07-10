import numpy as np
import crocoddyl
import torch
from tqdm import tqdm
from utils import Solver
from utils import Datagen
from utils import TerminalModelUnicycle
from network import ValueNet
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.set_default_dtype(torch.double)


class Irepa():
    def __init__(self, runs:int = 1, batch_size:int = 32, epochs:int = 250,
                lr:float = 1e-2, n_traj:int = 100, horizon:int = 30, weights:list = [1., 1.],
                precision:int = 1e-9, maxiters:int = 1000, fc1_dims:int = 256,
                fc2_dims:int = 256, fc3_dims:int = 120, weight_decay:float = 5e-4):
        """Irepa training"""

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

       

    def _train(self, neural_net, terminal_unicycle=None):
        """
        Args
        .........
                1: neural_net       : neural network to be trained
                2: starting_points  : trajectories will be generated from these starting points.
                3: terminal_unicycle: terminal model for irepa > 0

        Returns
        ........
                1: A trained neural net
        
        
        """
        ############################################################    Training        ####################################################################################################
        # Generate training data
        train_points = Datagen.gridData(n_points = self.n_traj)
        #train_points = Datagen.uniformSphere(n_points=self.n_traj, radius=1.9)
        #train_points = Datagen.random_positions(size=1000)
        x_train, y_train, = Solver(initial_configs=train_points,
                                   terminal_model=terminal_unicycle,
                                   weights=self.weights,
                                   horizon=self.horizon,
                                   precision=self.precision).solveNodesValues(as_tensor=True)

        #print("Training dataset size: ",x_train.shape[0])       
        
        validation_points = Datagen.gridData(n_points = 10, xy_limits=[-1.5, 1.5], theta_limits=[-.5, .5])
        #validation_points = Datagen.uniformSphere(n_points=20, radius=1.)

        x_valid, y_valid = Solver(initial_configs=validation_points,
                                 terminal_model=terminal_unicycle,
                                 weights=self.weights,
                                 horizon=20,
                                 precision=self.precision).solveNodesValues(as_tensor=True)
        #points = train_points
        points = x_train.numpy()
        points_cost = y_train.numpy()
        #points = x_valid.numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im = ax.scatter3D(points[:,0], points[:,1], points[:,2],c = points_cost, marker = ".", cmap="plasma")
        #plt.axis('off')
        ax.set_title("X_Train from Random Sampling")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)
        fig.colorbar(im).set_label("cost")
        plt.savefig("xgrid.png")
        plt.show()       
        """
        # Convert to torch dataloader
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True)
        
        
        criterion1 = torch.nn.MSELoss(reduction='mean')
        criterion2 = torch.nn.L1Loss(reduction='mean')

        #optimizer = torch.optim.SGD(neural_net.parameters(), lr = self.lr, momentum=0.6, weight_decay=self.weight_decay, nesterov=True)
        optimizer = torch.optim.Adam(neural_net.parameters(), lr = self.lr, betas=[0.5, 0.9], weight_decay=self.weight_decay)
        neural_net.train()
        for epoch in tqdm(range(self.epochs), ascii=True, desc="Training"):
            for data, target in dataloader:
                optimizer.zero_grad()
                output = neural_net(data)
                loss = criterion1(output, target) 
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                y_pred = neural_net(x_valid)
                error_mae = criterion2(y_pred, y_valid)
                error_mse = criterion1(y_pred, y_valid)

            print(f"Epoch : {epoch}, mse : {error_mse}, mae : {error_mae}")

        
        ############################################################   Validation        #######################################################################################

        
        # Validation
        print("Validation...")
        validation_points = Datagen.gridData(n_points = 10, xy_limits=[-1.5, 1.5], theta_limits=[-1., 1.])
        #validation_points = Datagen.uniformSphere(n_points=100, radius=1.9)

        x_valid, y_valid = Solver(initial_configs=validation_points,
                                 terminal_model=terminal_unicycle,
                                 weights=self.weights,
                                 horizon=20,
                                 precision=self.precision).solveNodesValues(as_tensor=True)
        
        
        loss_func1 = torch.nn.L1Loss(reduction='mean')
        loss_func2 = torch.nn.MSELoss(reduction='mean')
        

        y_pred = neural_net(x_valid)
        error_mae = loss_func1(y_pred, y_valid)
        error_mse = loss_func2(y_pred, y_valid)

        print(f"\n  MAE : {error_mae}")
        print(f"\n  MSE : {error_mse}")
        
        ########################################################## Stopping Criteria
        x0 = np.array([-.5, 1.5, 1])
        print("......\n")
        print("For starting position :", x0)
        terminal_ = TerminalModelUnicycle(neural_net=neural_net)
        ddp, log, = Solver(maxiters=1000, terminal_model=None)._solveProblem(initial_config=x0,logger=True)
        #print("Entering log for terminal")
        ddp2, log2 = Solver(terminal_model=terminal_, maxiters=1000)._solveProblem(initial_config=x0, logger=True)
        log1 = log.stops[1:]
        log2 = log2.stops[1:]
        pred = neural_net(torch.Tensor(x0)).detach().numpy().item()
        #print("\n Stopping Criteria:")
        #print(f"\n   Crocoddyl = {log1[1:5]}")
        #print(f"\n   Terminal Croc = {log2[1:5]}")

  
        vx = ddp.Vx[0]
        vx1 = ddp2.Vx[0]
        vx2 = neural_net.jacobian(torch.Tensor(x0)).detach().numpy()

        vxx = ddp.Vxx[0]
        vxx1 = ddp2.Vxx[0]
        vxx2 = neural_net.hessian(torch.Tensor(x0)).detach().numpy()
        print(f"\n ddp.cost: {ddp.cost}")
        print(f" Terminal_ddp.cost : {ddp2.cost}")
        print(f" Predicted : {pred}")

        print(f"\n ddp.Vx[0]: {vx}")
        print(f" Terminal_ddp.Vx[0]: {vx1}")
        print(f" Net Jacobian : {vx2}")

        #print(f"\n ddp.Vxx[0]: {vxx} || terminal_ddp.Vxx[0]: {vxx1} || Net Hessian : {vxx2}")
        plt.plot(log1, '--o', label="Crocoddyl")
        plt.plot(log2, "--o", label="Terminal")
        plt.legend()
        plt.show()

        
        del dataset, dataloader, x_valid, y_valid, y_pred, x_train, y_train
        return neural_net
        """

    def run(self):

        print(f"\n##################### Starting {self.runs} Irepa Iterations ###############################\n")
        start = time.time()
        

        net = ValueNet(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, fc3_dims=self.fc3_dims)
        print(net)

        for i in range(self.runs):
            print("\n.........\n")
            print(f"\n Iteration # {i}\n\n")
            if i == 0:
                terminal_model_unicycle = None
            else:
                terminal_model_unicycle = TerminalModelUnicycle(net)

            net = self._train(neural_net=net,terminal_unicycle=terminal_model_unicycle)
            #torch.save(net, f'./networks/exp2/net'+str(i)+'.pth')

        end = time.time()
        print("Total Time : ", end - start)






Irepa().run()





        