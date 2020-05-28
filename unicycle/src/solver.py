
"""
Solve a unicycle problem with crocoddyl and return ddp

"""
import numpy as np
import crocoddyl
import torch

def solve_problem(terminal_model=None,
                  initial_configuration=None,
                  horizon:int=30,
                  precision:float=1e-9,
                  maxiters:int=1000,
                  state_weight:float=1.,
                  control_weight:float=1.):
    """
    Solve the problem for a given initial_position.
    
    @params:
        1: terminal_model    = Terminal model with neural network inside it, for the crocoddyl problem.
                               If none, then Crocoddyl Action Model will be used as terminal model.
        
        2: initial_configuration = initial position for the unicycle, 
                                    either a list or a numpy array or a tensor.
        
        3: horizon           = Time horizon for the unicycle. Defaults to 30
        
        4: stop              = ddp.th_stop. Defaults to 1e-9
        
        5: maxiters          = maximum iterations allowed for crocoddyl.Defaults to 1000
                                
        5: state_weight      = defaults to 1.
        
        6: control_weight    = defaults to 1.
        
    @returns:
        ddp
    
    """
    if isinstance(initial_configuration, list):
        initial_configuration = np.array(initial_configuration)    
    
    elif isinstance(initial_configuration, torch.Tensor):
        initial_configuration = initial_configuration.numpy()

        
    model = crocoddyl.ActionModelUnicycle()
    model.costWeights = np.matrix([state_weight,control_weight]).T  
    
    if terminal_model is None:
        problem = crocoddyl.ShootingProblem(initial_configuration.T, [ model ] * horizon, model)
    else:
        problem = crocoddyl.ShootingProblem(initial_configuration.T, [ model ] * horizon, terminal_model)

        
    ddp         = crocoddyl.SolverDDP(problem)
    ddp.th_stop = precision
    
    ddp.solve([], [], maxiters)
    return ddp