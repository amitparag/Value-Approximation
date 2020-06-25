import numpy as np
import crocoddyl
import torch
from ddp_solver import solve_problem

class Datagen:
    
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

        #print(f"Sampling {size} [x, y, z] from: \n ")
        #print(f"  x = {xlim} \n")
        #print(f"  y = {ylim}\n")
        #print(f"  z = {zlim}\n")

        x = np.random.uniform(*xlim, size = (size, 1))
        y = np.random.uniform(*ylim, size = (size, 1))
        z = np.random.uniform(*zlim, size = (size, 1))
        
        dataset = np.hstack((x, y, z))
        
        if as_tensor:
            dataset = torch.tensor(dataset, dtype = torch.float32)
            return dataset
        
        else: return dataset
        
        
    def grid_data(size:int = 30,
                  limits = [-2., 2.]
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
        
        print(f" Returning {sum(n)} points from the circumference of a circle of radii {r}")
        
        circles = []
        for r, n in zip(r, n):
            t = np.linspace(0, 2* np.pi, n)
            x = r * np.cos(t)
            y = r * np.sin(t)
            z = np.zeros(x.size,)
            circles.append(np.c_[x, y, z])
        return np.array(circles).squeeze()
    
    
    def points(r=[2, 1, 0.5], n=[30, 40, 30]):
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
        angles = np.random.uniform(-2*np.pi, 2*np.pi, size = (size,1))
        data = np.vstack([i for i in np.array(data)])
        data = data[:,0:2]
        return np.hstack((data,angles))
    
    def data(ntrajectories = 100,
             xlim = [-2.1,2.1],
             ylim = [-2.1,2.1],
             zlim = [-2*np.pi,2*np.pi],
             terminal_model = None,
             horizon        = 100,
             precision      = 1e-9,
             maxiters       = 1000,
             state_weight   = 1.,
             control_weight = 1.,
             as_tensor      = True):
        """
        Generate the training and validation data for the unicycle of the form:
            positions --> values
        
        
        @params:
            1: ntrajectories  = number of trajectories to be generated.
                                The size of the dataset is going to be [ntrajectories X horizon]
                                
            2: xlim           = xlim of the positions to sample from
            3: ylim           = ylim of the positions to sample from  
            4: zlim           = zlim of the positions to sample from
            5: terminal_model = the terminal model to use for crocoddyl. If None, then ActionModelUnicycle is used.
            6: horizon        = time horizon, T
            7: precision      = ddp.th_stop, default value set to 1e-9. For validation use 1e-5
            8: maxiters       = maxiters for crocoddyl, 1000.
            9: state_weight   = state weight, default to 1
            10: control weight = control wight for the unicycle
            11: as_tensor      = True if tensor data is needed
        """
        #positions = Datagen.random_positions(size = ntrajectories,
        #                             xlim  = xlim,
        #                             ylim  = ylim,
        #                             zlim  = zlim)
        
        positions = Datagen.points()
        
        
        x_data = []
        y_data = []
        for position in positions:
            ddp = solve_problem(terminal_model = terminal_model,
                                initial_configuration = position,
                                horizon   = horizon,
                                precision = precision,
                                maxiters  = maxiters,
                                state_weight = state_weight,
                                control_weight = control_weight
                               )
            
            # Cost-to-go for every node in the horizon for the particular problem
            values = []
            for d in ddp.problem.runningDatas:
                values.append(d.cost)
                
            for i in range(len(values)):
                values[i] =  sum(values[i:]) + ddp.problem.terminalData.cost
            values.append(ddp.problem.terminalData.cost)
            xs = np.array(ddp.xs)
            
            for node, cost in zip(xs, values):
                x_data.append(node)
                y_data.append(cost)
            del values, xs    
                
        x_data, y_data = np.array(x_data), np.array(y_data).reshape(-1,1)
        
        if as_tensor:
            return torch.Tensor(x_data), torch.Tensor(y_data)
        
        else:
            return x_data, y_data
