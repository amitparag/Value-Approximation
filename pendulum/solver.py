"""
Solve a double problem with crocoddyl and return fddp
"""

import crocoddyl
import numpy as np
import torch
import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum

def solve_problem(terminal_model = None,
                  initial_configuration = None,
                  horizon:int = 100,
                  precision:float = 1e-9,
                  maxiters:int = 1000):
    
    
    """
    Solve the problem for a given initial_position.
    
    @params:
        1: terminal_model    = Terminal model with neural network inside it, for the crocoddyl problem.
                               If none, then Crocoddyl Integrated Action Model will be used as terminal model.
        
        2: initial_configuration = initial position for the unicycle, 
                                    either a list or a numpy array or a tensor.
        
        3: horizon           = Time horizon for the unicycle. Defaults to 100
        
        4: stop              = ddp.th_stop. Defaults to 1e-9
        
        5: maxiters          = maximum iterations allowed for crocoddyl.Defaults to 1000

        
    @returns:
        ddp
    """
    
    if isinstance(initial_configuration, list):
        initial_configuration = np.array(initial_configuration)    
    
    elif isinstance(initial_configuration, torch.Tensor):
        initial_configuration = initial_configuration.numpy()
        
        

    # Loading the double pendulum model
    robot = example_robot_data.loadDoublePendulum()
    robot_model = robot.model

    state = crocoddyl.StateMultibody(robot_model)
    actModel = ActuationModelDoublePendulum(state, actLink=1)

    weights = np.array([1, 1, 1, 1] + [0.1] * 2)
    runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
    xRegCost = crocoddyl.CostModelState(state, 
                                        crocoddyl.ActivationModelQuad(state.ndx),
                                        state.zero(),
                                        actModel.nu)
    
    uRegCost = crocoddyl.CostModelControl(state, 
                                          crocoddyl.ActivationModelQuad(1),
                                          actModel.nu)
    
    xPendCost = CostModelDoublePendulum(state, 
                                        crocoddyl.ActivationModelWeightedQuad(np.matrix(weights).T),
                                        actModel.nu)

    dt = 1e-2

    runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
    runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)


    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
    
    if terminal_model is None:
        terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)
        terminalCostModel.addCost("xGoal", xPendCost, 1e4)
        terminal_model = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)
        
    # Creating the shooting problem and the FDDP solver
    problem = crocoddyl.ShootingProblem(initial_configuration.T, [runningModel] * horizon, terminal_model)
    
    fddp = crocoddyl.SolverFDDP(problem)
    
    fddp.th_stop = precision
    
    fddp.solve([], [], maxiters)

    return fddp
    
        