


"""
Class to generate various kinds of datasets needed for experiments with unicycle

"""

import numpy as np
import crocoddyl
import torch
from solver import solve_problem


class Datagen:
    
    def values(positions,
               terminal_model = None,
               horizon        = 30,
               precision      = 1e-9,
               maxiters       = 1000,
               state_weight   = 1.,
               control_weight = 1.,
               as_tensor      = False):
        
        """
        Get the values after solving the problem with crocoddyl.
        
        @params:
            1: positions      = array or tensor of initial starting positions
            2: terminal_model = terminal model to be used while solving the problem.
                                If None, then it defaults to ActionModelUnicycle
                                
            3: horizon        = time horizon, T for the unicycle problem
            4: stop           = ddp.th_stop, default value --> 1e-9
                                For validation, etc, this should be set to 1e-5    
            5: maxiters       = Maximum iterations allowed for the unicycle problem
            6: state_weight   = Default to 1.
            7: control_weight = Default to 1.
            8: as_tensors     = type of dataset to return. If true, then the dataset returned will be a torch
                                tensor.
                                
        @returns:
        Solves the unicycle problem for each starting config in the positions data and returns ddp.cost
            1: values         =  array(or tensor) of ddp.cost   
        
        """
        
        
        values = []
        for position in positions:
            if isinstance(position, torch.Tensor):
                position = np.array(position.view(1, -1))
                
            ddp = solve_problem(terminal_model = terminal_model,
                                initial_configuration = position,
                                horizon = horizon,
                                precision = precision,
                                maxiters = maxiters,
                                state_weight = state_weight,
                                control_weight = control_weight)
            
            values.append([ddp.cost])
            
        if as_tensor:
            return torch.tensor(values, dtype = torch.float32)
        else:
            return np.array(values)
    
    
    
    def random_positions(size:int = 3000,
                         xlim = [-2.1,2.1],
                         ylim = [-2.1,2.1],
                         zlim = [-2*np.pi,2*np.pi],
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
        min_x, max_x = xlim
        min_y, max_y = ylim
        min_z, max_z = zlim

        print("Sampling x, y, z from: \n ")
        print(f"  x = [ {min_x} , {max_x} ] \n")
        print(f"  y = [ {min_y} , {max_y} ]\n")
        print(f"  z = [ {min_z} , {max_z} ]\n")

        x = np.random.uniform(min_x, max_x, size = (size, 1))
        y = np.random.uniform(min_y, max_y, size = (size, 1))
        z = np.random.uniform(min_z, max_z, size = (size, 1))
        
        dataset = np.hstack((x, y, z))
        
        if as_tensor:
            dataset = torch.tensor(dataset, dtype = torch.float32)
            return dataset
        
        else: return dataset
        
        
    def grid_data(size:int = 30,
                  limits = [-1., 1.]
                  ):

        """
        @params:
            1: size   = number of grid points
            2: limits = xlim, ylim
        
        @returns:
            1: grid array        
        """
        min_x, max_x = limits
        xrange = np.linspace(min_x,max_x,size)
        data = np.array([ [x1,x2, 0.] for x1 in xrange for x2 in xrange ])
        return data 
    
    
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
        
        print(f" Returning {n} points from the circumference of a circle of radii {r}")
        
        circles = []
        for r, n in zip(r, n):
            t = np.linspace(0, 2* np.pi, n)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = np.zeros(x.size,)
            circles.append(np.c_[x, y, z])
        return np.array(circles).squeeze()