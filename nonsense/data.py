"""
Data processing methods for crocoddyl

"""
import numpy as np
import torch
import crocoddyl
torch.set_default_dtype(torch.double)


class Solver():
    """
    This class will implement the following methods. 

    1: solveProblem
    2: nodesValues
    3: policiesValues
    
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
        yield ddp
    
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
            ddp = next(self.solveProblem(starting_config))

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

        # Shuffle the dataset
        permutation = np.random.permutation(values.shape[0])

        values = values[permutation]
        knots = knots[permutation]
        np.round_(values, 7)
        np.round_(knots, 7)
        if as_tensor:
            return torch.Tensor(knots), torch.Tensor(values)

        else:
            return knots, values


    def policiesValues(self, as_tensor:bool = True):
        """
        This method will return ddp.cost, ddp.xs and ddp.us for each starting configuration

        The dataset will be :
            xtrain  = starting_configs.
            ytrain1 = ddp.cost for all starting_configs.
            ytrain2 = each row will be ddp.xs[1:], flattened.
            ytrain3 = each row will be ddp.us[:], flattened.
        """

        xtrain  = []
        ytrain1 = []
        ytrain2 = []
        ytrain3 = []

        for starting_config in self.starting_configs:
            
            ddp = next(self.solveProblem(starting_config))
            xtrain.append(starting_config)
            ytrain1.append([ddp.cost])
            ytrain2.append(np.array(ddp.xs[1:]).flatten().tolist())
            ytrain3.append(np.array(ddp.us).flatten().tolist())
            del ddp

        xtrain  = np.array(xtrain)
        ytrain1 = np.array(ytrain1).reshape(-1, 1)
        ytrain2 = np.array(ytrain2)
        ytrain3 = np.array(ytrain3)

        np.round_(xtrain, 7)
        np.round_(ytrain1, 7)
        np.round_(ytrain2, 7)
        np.round_(ytrain3, 7)



        # Shuffle the dataset
        permutation = np.random.permutation(xtrain.shape[0])

        xtrain  = xtrain[permutation]
        ytrain1 = ytrain1[permutation]
        ytrain2 = ytrain2[permutation]
        ytrain3 = ytrain3[permutation]

        if as_tensor:
            xtrain  = torch.Tensor(xtrain)
            ytrain1 = torch.Tensor(ytrain1)
            ytrain2 = torch.Tensor(ytrain2)
            ytrain3 = torch.Tensor(ytrain2)
        return xtrain, ytrain1, ytrain2, ytrain3



class Datagen:
    def griddedData(n_points:int = 150,
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


points = Datagen.griddedData()
print(points)