"""
Data processing methods for crocoddyl

"""
import numpy as np
import torch
import crocoddyl
torch.set_default_dtype(torch.double)


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

    def uniformSphere(n_points:int = 10, radius:float = 1.9):
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
        return np.array(points)




class Solver():

    """
    This class will implement the following methods. 

    1: solveProblem
    2: nodesValues
    3: optimalStateTrajectory
    
    """
    def __init__(self,
                starting_configs:np.ndarray, 
                terminal_model = None,
                weights:list = [1., 1.], 
                horizon:int = 100,
                precision:float = 1e-9,
                maxiters:int = 1000
                ):


        
        """
        Solve unicycle problems with the given params

        Args
        .........

        1: starting_configs         = starting positions for the unicycle problem
        2: terminal_model           = terminal model for the unicycle problem
        3: weights                  = state and control weights
        4: horizon                  = time horizon for the problem
        5: precision                = ddp.th_stop
        6: maxiters                 = maximum iterations allowed
           
        
        """
        self.starting_configs = starting_configs
        self.terminal_model   = terminal_model
        self.weights          = weights
        self.horizon          = horizon
        self.precision        = precision
        self.maxiters         = maxiters

        self.starting_configs = np.atleast_2d(self.starting_configs)

            

    def solveProblem(self, starting_config):
        """Solve unicycle problems and yield ddp """
        model = crocoddyl.ActionModelUnicycle()
        if self.terminal_model is not None:
            terminal_unicycle = self.terminal_model
        else:
            terminal_unicycle = crocoddyl.ActionModelUnicycle()
        assert terminal_unicycle is not None
        model.costWeights = np.array([*self.weights]).T
        problem = crocoddyl.ShootingProblem(starting_config, [model] * self.horizon, terminal_unicycle)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.th_stop = self.precision
        ddp.solve([], [], self.maxiters)
        del model, terminal_unicycle, problem
        return ddp
    
    def nodesValues(self, as_tensor:bool = True):

        """
        Return nodes and cost associated with each node in the trajectory for each trajectory generated from the starting configs.
        
        This method calculates the optimal cost from each node in a trajectory, and returns the knots and values after shuffling them.
        The dataset thus generated will look like:
        starting_positions = xtrain
        optimal_cost       = ytrain
        """
        
        knots   = []
        values  = []

        for starting_config in self.starting_configs:
            ddp = self.solveProblem(starting_config)

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
        del cost, ddp
        
        knots = np.array(knots)
        values = np.array(values).reshape(-1, 1)


        if as_tensor:
            return torch.Tensor(knots), torch.Tensor(values)

        else:

            return knots, values

    def optimalStateTrajectory(self, as_tensor:bool = True):
        """
        Given the starting configs, solve for optimal state trajectory.
        The dataset will be :
            xtrain  = starting_configs.
            ytrain = each row will be ddp.xs[1:], flattened.
        
        """
        xtrain = []
        ytrain = []

        for starting_config in self.starting_configs:
            
            ddp = self.solveProblem(starting_config)
            xtrain.append(starting_config)
            xs = np.array(ddp.xs)
            xs = np.array(xs[1:,:]).tolist()
            ytrain.append(xs)
        del ddp
        xtrain  = np.array(xtrain)
        ytrain = np.array(ytrain)

        if as_tensor:
            xtrain  = torch.Tensor(xtrain)
            ytrain = torch.Tensor(ytrain)

        return xtrain, ytrain


if __name__=="__main__":

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    starting_points = Datagen.griddedData(1000)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(starting_points[:,0], starting_points[:,1], starting_points[:,2])
    plt.show()