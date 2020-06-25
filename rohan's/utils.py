
import numpy as np
import crocoddyl
import torch

def solve_problem(terminal_model = None,
                  initial_configuration = None,
                  horizon:int = 100,
                  precision:float = 1e-9,
                  maxiters:int = 1000,
                  weights:list = [1., 1.]):
    """
    Solve the problem for a given initial_position.
    
    @params:
        1: terminal_model    = Terminal model with neural network inside it, for the crocoddyl problem.
                               If none, then Crocoddyl Action Model will be used as terminal model.
        
        2: initial_configuration = initial position for the unicycle, 
                                    either a list or a numpy array or a tensor.
        
        3: horizon           = Time horizon for the unicycle. Defaults to 100
        
        4: stop              = ddp.th_stop. Defaults to 1e-9
        
        5: maxiters          = maximum iterations allowed for crocoddyl.Defaults to 1000
                                
        5: weights           = state and control weights. defaults to 1.
  
        
    @returns:
        ddp
    
    """
    if isinstance(initial_configuration, list):
        initial_configuration = np.array(initial_configuration)    
    
    elif isinstance(initial_configuration, torch.Tensor):
        initial_configuration = initial_configuration.numpy()

        
    model = crocoddyl.ActionModelUnicycle()
    model.costWeights = np.array([*weights]).T  
    
    if terminal_model is None:
        problem = crocoddyl.ShootingProblem(initial_configuration.T, [ model ] * horizon, model)
    else:
        problem = crocoddyl.ShootingProblem(initial_configuration.T, [ model ] * horizon, terminal_model)

        
    ddp         = crocoddyl.SolverDDP(problem)
    ddp.th_stop = precision
    
    ddp.solve([], [], maxiters)
    return ddp


def random_positions(size:int = 3000,
                         xlim = [-2.1,2.1],
                         ylim = [-2.1,2.1],
                         zlim = [-np.pi,np.pi],
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


def grid_data(size:int = 10,
             xy_limits = [-2., 2.],
             theta_lims = [-np.pi/2, np.pi/2],
             as_tensor:bool = False
             ):
    """
    @params:
        1: size   = number of grid points
        2: limits = xlim, ylim
    
    @returns:
        1: grid array        
    """
    min_x, max_x = xy_limits
    xrange = np.linspace(min_x,max_x,size)
    dataset = np.array([ [x1,x2, np.random.uniform(*theta_lims)] for x1 in xrange for x2 in xrange ])
    
    if as_tensor:
        dataset = torch.tensor(dataset, dtype = torch.float32)
        return dataset
    
    else: return dataset

    
def circular_data(r=[2], n=[100]):
    """
    @params:
        r = list of radii
        n = list of number of points required from each radii
        
    @returns:
        array of points from the circumference of circle of radius r centered on origin
        
    Usage: circle_points([2, 1, 3], [100, 20, 40]). This will return 100, 20, 40 points from
            circles of radius 2, 1 and 3
    """
    
    print(f" Returning {sum(n)} points from the circumference of a circle of radii {r}")
    
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2* np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros(x.size,)
        circles.append(np.c_[x, y, z])
    return np.array(circles).squeeze()


def points(r=[2, 1, 0.5], n=[33, 40, 33]):
    """
    @params:
        r = list of radii
        n = list of number of points required from each radii

    @returns:
        array of points from the circumference of circle of radius r centered on origin

    Usage: circle_points([2, 1, 3], [100, 20, 40]). This will return 100, 20, 40 points from
            circles of radius 2, 1 and 3
    """

    print(f" Returning {sum(n)} points from the circumference of a circle of radii {r}")
    size = sum(n)

    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 2* np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = np.zeros(x.size,)
        circles.append(np.c_[x, y, z])
    data = np.array(circles).squeeze()
    angles = np.random.uniform(-np.pi/2, np.pi/2, size = (size,1))
    data = np.vstack([i for i in np.array(data)])
    data = data[:,0:2]
    return np.hstack((data,angles))

def statesValues(init_positions,terminal_model=None, horizon=100, precision=1e-9,weights=[1., 1], maxiters = 1000,as_tensor:bool = True ):
    x_data = []
    y_data = []
    for position in init_positions:
        ddp = solve_problem(terminal_model = terminal_model,
                            initial_configuration = position,
                            horizon   = horizon,
                            precision = precision,
                            maxiters  = maxiters,
                            weights   = weights)
        
        # Cost-to-go for every node in the horizon for the particular problem
        values = []
        for d in ddp.problem.runningDatas:
            values.append(d.cost)
            
        for i in range(len(values)):
            values[i] =  sum(values[i:]) + ddp.problem.terminalData.cost
        values.append(ddp.problem.terminalData.cost)    
        xs = np.array(ddp.xs)
        assert xs.shape[0] == len(values) == horizon +1
        for node, cost in zip(xs, values):
            x_data.append(node)
            y_data.append(cost)
        del values, xs    
            
    x_data, y_data = np.array(x_data), np.array(y_data).reshape(-1,1)
    
    if as_tensor:
        return torch.tensor(x_data, dtype = torch.float32), torch.tensor(y_data, dtype = torch.float32)
    
    else:
        return x_data, y_data


