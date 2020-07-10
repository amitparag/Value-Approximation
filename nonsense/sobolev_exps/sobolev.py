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


class Sobolev():
    def __init__(self, runs:int = 1, batch_size:int = 32, epochs:int = 250,
                lr:float = 0.001, n_traj:int = 100, horizon:int = 50, weights:list = [1., 1.],
                precision:int = 1e-7, maxiters:int = 1000, fc1_dims:int = 20,
                fc2_dims:int = 20, fc3_dims:int = 2, weight_decay:float = 0.):
        """Sobolev training"""

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
        starting_points = Datagen.gridData(n_points = self.n_traj)
        x_train, y_train1, y_train2 = Solver(initial_configs=starting_points, 
                                            terminal_model=terminal_unicycle,
                                            weights=self.weights,
                                            horizon=self.horizon,
                                            precision=self.precision).solveNodesValuesGrads(as_tensor=True)


        grads = y_train2.detach().numpy()
        cost = y_train1.detach().numpy()
        positions = x_train.detach().numpy()


        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im2 = ax.scatter3D(grads[:,0], grads[:,1], grads[:,2],c = cost, marker = ".", cmap="viridis")
        ax.set_title("Grad input to Neural Net")
        ax.set_xlabel("Grad[0]")
        ax.set_ylabel("Grad[1]")
        ax.set_zlabel("Grad[2]")
        ax.grid(False)
        ax.grid(False)
        fig.colorbar(im2).set_label("cost")

        plt.savefig("GradInput.png")
        plt.show()


        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im2 = ax.scatter3D(positions[:,0], positions[:,1], positions[:,2],c = cost, marker = ".", cmap="viridis")
        ax.set_title("Xtrain, Ytrain for Neural Net")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)
        fig.colorbar(im2).set_label("cost")

        plt.savefig("StateSpaceInput.png")

        plt.show()

        del grads, cost, positions
















                
        x_valid, y_valid = x_train[0:20, :], y_train1[0:20,:]

        x_train = x_train[20:, :]
        y_train1 = y_train1[20:, :]
        y_train2 = y_train2[20:, :]



        # Convert to torch dataloader
        dataset = torch.utils.data.TensorDataset(x_train, y_train1, y_train2)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True)
        
        
        criterion1 = torch.nn.MSELoss(reduction='sum')
        criterion2 = torch.nn.L1Loss(reduction='sum')
        
        
        criterion3 = torch.nn.MSELoss(reduction='mean')
        criterion4 = torch.nn.L1Loss(reduction='mean')
        #optimizer = torch.optim.SGD(neural_net.parameters(), lr = self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)
        #optimizer = torch.optim.Adam(neural_net.parameters(), lr = self.lr, betas=[0.5, 0.9], weight_decay=self.weight_decay,amsgrad=True)
        optimizer = torch.optim.ASGD(neural_net.parameters(), lr = self.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=self.weight_decay)


        neural_net.train()
        for epoch in range(self.epochs):
            neural_net.train()

            for data, target1, target2 in dataloader:
                #data.requires_grad=True
                optimizer.zero_grad()
                output1 = neural_net(data)
                output2 = neural_net.batch_jacobian(data)
                loss = criterion2(output1, target1) + 0.001*criterion1(output2, target2)
                
                loss.backward()
                optimizer.step()

            with torch.no_grad():

                y_pred = neural_net(x_valid)
                error_mse = criterion3(y_pred, y_valid)
                error_mae = criterion4(y_pred, y_valid)

            print(f"Epoch {epoch + 1} :: mse = {error_mse}, mae = {error_mae}")
        
        
        

        ### Let's solve a problem

        x0 = np.array([-.5, .5, 0.34]).reshape(-1, 1)
        
        model = crocoddyl.ActionModelUnicycle()
        model2 = crocoddyl.ActionModelUnicycle()
        model.costWeights = np.array([1., 1.]).T
        model2.costWeights = np.array([1., 1.]).T
        terminal_= TerminalModelUnicycle(neural_net)

        problem1 = crocoddyl.ShootingProblem(x0, [model] * 20, model)
        problem2 = crocoddyl.ShootingProblem(x0, [model] * 20, terminal_)

        ddp1 = crocoddyl.SolverDDP(problem1)
        ddp2 = crocoddyl.SolverDDP(problem2)

        log1 = crocoddyl.CallbackLogger()
        log2 = crocoddyl.CallbackLogger()

        ddp1.setCallbacks([log1])
        ddp2.setCallbacks([log2])

        ddp1.solve([], [], 1000)
        ddp2.solve([], [], 1000)

        with torch.no_grad():
            predict = neural_net(torch.Tensor(x0.T)).detach().numpy().item()

        print("\n ddp_c :", ddp1.cost)
        print("\n ddp_t :", ddp2.cost)
        print("\n net   :", predict)


        
        plt.clf()
        plt.plot(log1.stops[1:], '--o', label="C")
        plt.plot(log2.stops[1:], '--.', label="T")
        plt.xlabel("Iterations")
        plt.ylabel("Stopping Criteria")
        plt.title(f"ddp.cost: {ddp1.cost}, t_ddp.cost: {ddp2.cost}, net_pred_cost: {predict}")
        plt.legend()
        plt.savefig("sc.png")
        plt.show()

        starting_points2 = Datagen.gridData(n_points = self.n_traj, xy_limits=[-1, 1], theta_limits=[-0.5, 0.5])
        x_test, y_test1, y_test2 = Solver(initial_configs=starting_points2, 
                                            terminal_model=terminal_unicycle,
                                            weights=self.weights,
                                            horizon=self.horizon,
                                            precision=self.precision).solveNodesValuesGrads(as_tensor=True)

        
        
        
        with torch.no_grad():
            pred_cost = neural_net(x_test).detach().numpy()

        grads_pred = neural_net.batch_jacobian(x_test).detach().numpy()
        x_test = x_test.detach().numpy()
        
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        im1 = ax.scatter3D(x_test[:,0], x_test[:,1], x_test[:,2],c = pred_cost, marker = ".", cmap="viridis")
        #plt.axis('off')
        ax.set_title("Prediction")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.grid(False)
        fig.colorbar(im1).set_label("cost")
        plt.savefig('predicted.png')
        plt.show()

        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        im2 = ax.scatter3D(grads_pred[:,0], grads_pred[:,1], grads_pred[:,2],c = pred_cost, marker = ".", cmap="viridis")
        ax.set_title("jacobian Net")
        ax.set_xlabel("Grad[0]")
        ax.set_ylabel("Grad[1]")
        ax.set_zlabel("Grad[2]")
        ax.grid(False)
        ax.grid(False)
        fig.colorbar(im2).set_label("cost")
        plt.savefig('predicted_grads.png')
        plt.show()
        

        del dataset, dataloader, x_valid, y_valid, y_pred, x_train, y_train1, y_train2, x_test, y_test1, y_test2
        return neural_net


    def run(self):

        print(f"\n##################### Starting {self.runs} Sobolev Iterations ###############################\n")

        

        net = ValueNet(fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, fc3_dims=self.fc3_dims)
        print(net)

        start = time.time()


        for i in range(self.runs):
            print("\n.........\n")
            print(f"\n Iteration # {i}\n\n")
            if i == 0:
                terminal_model_unicycle = None
            else:
                terminal_model_unicycle = TerminalModelUnicycle(net)
            net = self._train(neural_net=net, terminal_unicycle=terminal_model_unicycle)
            #torch.save(net, f'./networks/exp1/net'+str(i)+'.pth')

        end = time.time()
        print("Total Time : ", end - start)           




if __name__=='__main__':
    Sobolev().run()





        