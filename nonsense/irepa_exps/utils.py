"""
This contains two classes Solver and TerminalModelUnicycle

"""

import numpy as np
import crocoddyl
import torch
import random
torch.set_default_dtype(torch.double)


class Solver():
    def __init__(self, initial_configs:np.ndarray = None, terminal_model = None, weights:list = [1., 1.], 
                horizon:int = 50, precision:float = 1e-9, maxiters:int = 1000):

        """
        Solve unicycle problems with the given params

        Args
        .........

        1: initial_configs          = starting positions for the unicycle problem
        2: terminal_model           = terminal model for the unicycle problem
        3: weights                  = state and control weights
        4: horizon                  = time horizon for the problem
        5: precision                = ddp.th_stop
        6: maxiters                 = maximum iterations allowed
           
        
        """
        self.initial_configs = initial_configs
        self.terminal_model  = terminal_model
        self.weights         = weights
        self.horizon         = horizon
        self.precision       = precision
        self.maxiters        = maxiters
        
    def _solveProblem(self, initial_config, logger:bool = False):
        """Solve one unicycle problem"""

        if not isinstance(initial_config, np.ndarray):
            initial_config = np.array(initial_config).reshape(-1, 1)

        model = crocoddyl.ActionModelUnicycle()
        model.costWeights = np.array([*self.weights]).T
        
        if self.terminal_model is None:
            problem = crocoddyl.ShootingProblem(initial_config, [model] * self.horizon, model)
        else:
            problem = crocoddyl.ShootingProblem(initial_config, [model] * self.horizon, self.terminal_model)

        ddp = crocoddyl.SolverDDP(problem)
        if logger:
            log = crocoddyl.CallbackLogger()
            ddp.setCallbacks([log])

        ddp.th_stop = self.precision
        ddp.solve([], [], self.maxiters)

        if logger:
            #print("\n Returning logs and ddp")
            return ddp, log

        else: return ddp

    def solveProblems(self):
        assert self.initial_configs is not None

        self.initial_configs = np.atleast_2d(self.initial_configs)
        
        ddp_solutions = []
        
        for initial_config in self.initial_configs:
            ddp = self._solveProblem(initial_config=initial_config, logger=False)
            ddp_solutions.append(ddp)
        
        return ddp_solutions


    def solveNodesValues(self, as_tensor:bool = True):
        """Get the nodes and the corresponding values from all trajectories generated."""
        ddps = self.solveProblems()

        knots, values = [], []

        for ddp in ddps:
            # Append nodes to knots
            xs = np.array(ddp.xs).tolist()
            for node in xs:
                knots.append(node)

            cost = []

            for d in ddp.problem.runningDatas:
                cost.append(d.cost)
            cost.append(ddp.problem.terminalData.cost)

            for i, _ in enumerate(cost):
                cost[i] = sum(cost[i:])

            # Append costs in cost to values
            for c in cost:
                values.append(c)

        values = np.array(values).reshape(-1, 1)
        knots = np.array(knots)
        np.round_(values, decimals=6)
        np.round_(knots, decimals=6)
        del cost

        if as_tensor:
            dataset = np.hstack((knots, values))
            np.random.shuffle(dataset)

            x_train = dataset[:,0:-1]
            y_train = dataset[:,-1]
            y_train = y_train.reshape(-1, 1)

            assert x_train.ndim == y_train.ndim

            return torch.Tensor(x_train), torch.Tensor(y_train)
        else: return knots, values


class TerminalModelUnicycle(crocoddyl.ActionModelAbstract):
    """
    This includes a feedforward network in crocoddyl
    
    """
    def __init__(self, neural_net):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = neural_net

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        np.round_(x, decimals=6)    
        x = torch.Tensor(x).resize_(1, 3)
        # Get the cost
        with torch.no_grad():
            data.cost = self.net(x).item()


    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        np.round_(x, decimals=6)    
    
        # This is irritating. Converting numpy to torch everytime.
        x = torch.Tensor(x).resize_(1, 3)
        
        data.Lx =j = self.net.jacobian(x).detach().numpy()
        
        data.Lxx = self.net.hessian(x).detach().numpy()


class Datagen:
    def uniformSphere(n_points:int = 100, radius:float = 1.9):
        """Sample uniformly from a cube of given radius"""

        points = []

        for _ in range(n_points):
            u = np.random.random()
            x1 = np.random.uniform(-2, 2)
            x2 = np.random.uniform(-2., 2.)
            x3 = np.random.uniform(-np.pi/2, np.pi/2)
            
            mag = np.sqrt(x1*x1 + x2*x2 + x3*x3)
            x1 /= mag
            x2 /= mag
            x3 /= mag
            
            c = radius * np.cbrt(u)

            point = [x1*c, x2*c, x3*c]

            points.append(point)

        points = np.array(points)
        np.round_(points, 6)
        np.random.shuffle(points)

        return points



    def gridData(n_points:int = 100, xy_limits:list = [-2.0, 2.0], theta_limits:list = [-np.pi/2, np.pi/2] ):
        size = int(np.sqrt(n_points)) + 1

        min_x, max_x = [*xy_limits]
        xrange = np.linspace(min_x,max_x,size, endpoint=True)
        trange = np.linspace(*theta_limits, size, endpoint=True)
        points = np.array([ [x1,x2, x3] for x1 in xrange for x2 in xrange for x3 in trange])

        np.round_(points, decimals=6)
        np.random.shuffle(points)
        points = points[0:n_points, : ]
        return points


    def random_positions(size:int = 100,
                         xlim = [-2.1,2.1],
                         ylim = [-2.1,2.1],
                         zlim = [-np.pi/2,np.pi/2],
                         as_tensor:bool = False):
        """
        Generate randomly sampled x, y, z from the ranges given.
        @params:
            1: size      = size of the array to generate
            2: xlim      = range of x positions
            3: ylim      = range of y positions
            4: zlim      = range of z positions
            5: as_tensor = bool, True if data is needed in the form of tensors
            
        @returns:
            1: dataset = [x, y, theta], where x, y, theta have been generated randomly
        
        """

        x = np.random.uniform(*xlim, size = (size, 1))
        y = np.random.uniform(*ylim, size = (size, 1))
        z = np.random.uniform(*zlim, size = (size, 1))
        
        dataset = np.hstack((x, y, z))
        
        if as_tensor:
            dataset = torch.tensor(dataset, dtype = torch.float32)
            return dataset
        
        else: return dataset

if __name__=="__main__":

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    points = Datagen.gridData(xy_limits=[-2, 2], theta_limits=[-np.pi/2, np.pi/2])
    #oints = Datagen.gridData()
    #points = Datagen.uniformSphere()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(points[:,0], points[:,1], points[:,2], marker = ".",cmap="plasma", )
    #plt.axis('off')
    ax.set_title(" Sampling from a Grid")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(False)
    plt.savefig("grid.png")
    plt.show()

