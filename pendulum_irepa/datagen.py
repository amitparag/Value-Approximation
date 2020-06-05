import numpy as np
import crocoddyl
import torch
from ddp_solver import solve_problem

class Datagen:
    
    
    def values(init_conitions,
               terminal_model = None,
               horizon        = 100,
               precision      = 1e-9,
               maxiters       = 1000,
               as_tensor      = False):
        """
        Get the values after solving the problem with crocoddyl.
        
        @params:
            1: init_conditions = array or tensor of initial starting positions
            2: terminal_model  = terminal model to be used while solving the problem.
                                 If None, then defaults to IntegratedActionModel
                                                                 
            3: horizon         = time horizon, T 
            4: stop            = ddp.th_stop, default value --> 1e-9
                                 For validation, etc, this should be set to 1e-5    
            5: maxiters        = Maximum iterations allowed 

            8: as_tensors      = type of dataset to return. If true, then the dataset returned will be a torch
                                 tensor.
                                
        @returns:
        Solves the problem for each starting config in the positions data and returns ddp.cost
            1: values         =  array(or tensor) of ddp.cost   
            
        """
        values = []
        for initial_cond in init_conitions:
            ddp = solve_problem(terminal_model = terminal_model,
                                initial_configuration = initial_cond,
                                horizon = horizon,
                                precision = precision,
                                maxiters = maxiters)
            values.append([ddp.cost])
            
        if as_tensor:
            return torch.Tensor(values)
        else:
            return np.array(values)

            
    
    def random_starting_conditions(size:int   = 50,
                                   angle1_lim = [-2.*np.pi, 2.*np.pi],
                                   angle2_lim = [-2.*np.pi, 2.*np.pi],
                                   vel1_lim   = [-1., 1.],
                                   vel2_lim   = [-1., 1.],
                                   as_tensor  = True
                                   ):
        """
        Generate an array of random starting conditions for the double pendulum.
        The fddp solver takes 4 params as input: theta1, theta2, velocity1, velocity2

        @params
                1: size       = size of the array to generate
                2: angle1_lim = range of theta1
                3: angle1_lim = range of theta2
                4: vel1_lim   = range of velocity1
                5: vel2_lim   = range of velocity2
                5: as_tensor  = bool, True if data is needed in the form of tensors


        @returns:
                1: dataset = [angle1, angle2, velocity1, velocity2]
        """

        print("Sampling:\n ")
        print(f"theta1 from {angle1_lim}")
        print(f"theta2 from {angle2_lim}")
        print(f"vel1   from {vel1_lim}")
        print(f"vel2   from {vel2_lim}")
        
        theta1 = np.random.uniform(*angle1_lim, size = (size, 1))
        theta2 = np.random.uniform(*angle2_lim, size = (size, 1))
        vel1   = np.random.uniform(*vel1_lim, size = (size, 1))
        vel2   = np.random.uniform(*vel2_lim, size = (size, 1))
        
        dataset = np.hstack((theta1, theta2, vel1, vel2))

        
        if as_tensor:
            dataset = torch.Tensor(dataset)
            return dataset
        
        else: return dataset
        
        
    def grid_data(size:int = 30,
                  limits = [-2.*np.pi, 2.*np.pi]
                  ):

        """
        @params:
            1: size   = number of grid points
            2: limits = theta1_lim, theta2_lim
        
        @returns:
            1: grid array        
        """
        min_x, max_x = limits
        xrange = np.linspace(min_x,max_x,size)
        data = np.array([ [x1,x2, 0., 0.] for x1 in xrange for x2 in xrange ])
        return data 
        
