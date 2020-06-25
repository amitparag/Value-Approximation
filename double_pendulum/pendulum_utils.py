
import numpy as np
import crocoddyl
import torch
import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
from typing import Union
from neural_net import feedForwardNet
torch.set_default_tensor_type(torch.DoubleTensor)

class Solver:
    """Class to get ddp solutions for a given terminal_model and position"""
    
    def optimalSolution(init_states : Union[list, np.ndarray, torch.Tensor] ,
                        terminal_model : crocoddyl.ActionModelAbstract = None,
                        horizon : int = 150,
                        precision : float = 1e-9,
                        maxiters : int = 1000,
                        use_fddp : bool = True):
        
        """Solve double pendulum problem with the given terminal model for the given position
        
        Parameters
        ----------
        init_states   : list or array or tensor
                            These are the initial, starting configurations for the double pendulum
        
        terminal_model: crocoddyl.ActionModelAbstract
                            The terminal model to be used to solve the problem
                            
        horizon       : int
                            Time horizon for the running model
                            
        precision     : float
                            precision for ddp.th_stop
                            
        maxiters      : int
                            Maximum iterations allowed for the problem
                            
        use_fddp      : boolean
                            Solve using ddp or fddp
        
        Returns
        --------
        
        ddp           : crocoddyl.Solverddp
                            the optimal ddp or fddp of the prblem
        """
        
        if isinstance(init_states, torch.Tensor):
            init_states = init_states.numpy()
        init_states = np.atleast_2d(init_states)
        
        solutions = []
        
        for init_state in init_states:
            robot = example_robot_data.loadDoublePendulum()
            robot_model = robot.model

            state = crocoddyl.StateMultibody(robot_model)
            actModel = ActuationModelDoublePendulum(state, actLink=1)

            weights = np.array([1.5, 1.5, 1, 1] + [0.1] * 2)
            runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
            dt = 1e-2

            xRegCost = crocoddyl.CostModelState(state, 
                                                crocoddyl.ActivationModelQuad(state.ndx),
                                                state.zero(),
                                                actModel.nu)

            uRegCost = crocoddyl.CostModelControl(state,
                                                  crocoddyl.ActivationModelQuad(1),
                                                  actModel.nu)
            xPendCost = CostModelDoublePendulum(state, 
                                                crocoddyl.ActivationModelWeightedQuad(weights),
                                                actModel.nu)

            runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
            runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)

            runningModel = crocoddyl.IntegratedActionModelEuler(
                                                            crocoddyl.DifferentialActionModelFreeFwdDynamics(state,
                                                            actModel,
                                                            runningCostModel),
                                                            dt)

            if terminal_model is None:
                terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)
                terminalCostModel.addCost("xGoal", xPendCost, 1e4)
                terminal_model = crocoddyl.IntegratedActionModelEuler(
                        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel,
                                                                         terminalCostModel),
                                                                         dt)


            problem = crocoddyl.ShootingProblem(init_state, [runningModel] * horizon, terminal_model)
            if use_fddp:
                fddp = crocoddyl.SolverFDDP(problem)
            else:
                fddp = crocoddyl.SolverDDP(problem)

            fddp.th_stop = precision

            fddp.solve([], [], maxiters)

            solutions.append(fddp)
        return solutions
    
    
    
class Datagen:
    """Class to do various data generation processes needed to run IREPA."""

    def samples(size: int = 10,
                sampling: str = 'uniform',
                th1_lims : list = [-2*np.pi, 2*np.pi],
                th2_lims : list = [-2*np.pi, 2*np.pi],
                vel1_lims : list = [-0.5, 0.5],
                vel2_lims : list = [-0.5, 0.5],
                as_tensor : bool = True):
        
        """Generate either uniform or grid samples from the given limits"""
        
        if sampling == 'uniform':
                     
            theta1 = np.random.uniform(*th1_lims, size).reshape(size, 1)
            theta2 = np.random.uniform(*th2_lims, size).reshape(size, 1)
            vel1 = np.random.uniform(*vel1_lims, size).reshape(size, 1)
            vel2 = np.random.uniform(*vel2_lims, size).reshape(size, 1)

            samples = np.hstack((theta1, theta2, vel1, vel2))
            
        elif sampling == 'grid':
            xrange = np.linspace(*th1_lims,size)
            vrange = np.linspace(*vel1_lims, size)
            samples = np.array([ [x1,x2,v1,v2] for x1 in xrange for x2 in xrange
                                for v1 in vrange for v2 in vrange])
        else:
            print("Error in sampling")
            
        if as_tensor:
            samples = torch.tensor(samples, dtype = torch.float64)
        return samples

        
    
    def statesValues(init_states : Union[list, np.ndarray, torch.Tensor],
                     terminal_model : crocoddyl.ActionModelAbstract = None,
                     horizon : int = 250,
                     precision : float = 1e-9,
                     maxiters : int = 1000,
                     use_fddp : bool = True,
                     full_traj : bool = False,
                     as_tensor : bool = True
                     ):
        """Solves double pendulum problem for the given init states and returns states (along the 
        trajectory) with their corresponding values"""
        
        if full_traj:
            state_space = []
            
        values = []
        # Get the ddp solutions for the init states
        ddp_solutions = Solver.optimalSolution(init_states=init_states,
                                               terminal_model= terminal_model,
                                               horizon = horizon,
                                               precision = precision,
                                               maxiters = maxiters,
                                               use_fddp = use_fddp)
        
        if not full_traj:
            values.append([ddp.cost for ddp in ddp_solutions])
            values = np.array(values).reshape(init_states.shape[0], 1)
            if as_tensor:
                return torch.tensor(values)
            else: return values
            
        else:
            for ddp in ddp_solutions:
                xs_states = np.array(ddp.xs)
                xs_states_cost = []
                for d in ddp.problem.runningDatas:
                    xs_states_cost.append(d.cost)
                
                for i in range(len(xs_states_cost)):
                    xs_states_cost[i] =  sum(xs_states_cost[i:]) + ddp.problem.terminalData.cost
                xs_states_cost.append(ddp.problem.terminalData.cost)
                
                for node, cost in zip(xs_states, xs_states_cost):
                    state_space.append(node)
                    values.append(cost)
            
            state_space = np.array(state_space)
            values      = np.array(values).reshape(-1,1)
            assert state_space.shape[0] == values.shape[0]
            
            if as_tensor: return torch.tensor(state_space, dtype = torch.float64), torch.tensor(values, dtype=torch.float64)
            else: return state_space, values
                
                
                
        
class terminalPendulum(crocoddyl.ActionModelAbstract):
    """
    This includes a feedforward network in crocoddyl
    
    """
    def __init__(self, neural_net, robot):
        crocoddyl.ActionModelAbstract.__init__(self,
                                               crocoddyl.StateMultibody(robot),
                                               0,0)
        
        self.net = neural_net
        self.net.double()
    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
            
        x = torch.tensor(x, dtype=torch.float64).resize_(1, 4)
        
        # Get the cost
        with torch.no_grad():
            data.cost = self.net(x).item()
            
    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
            
        # This is irritating. Converting numpy to torch everytime.
        x = torch.tensor(x, dtype=torch.float64).resize_(1, 4)
        
        data.Lx = self.net.jacobian(x).detach().numpy()
        data.Lxx = self.net.hessian(x).detach().numpy()
            
    
                         
