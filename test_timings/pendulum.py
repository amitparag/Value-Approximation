import numpy as np
import example_robot_data
import crocoddyl
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
from time import perf_counter

def solver(starting_condition, T = 30, precision = 1e-9, maxiters = 1000):
    """Solve one pendulum problem"""
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
                                                    runningCostModel), dt)

    terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)
    terminalCostModel.addCost("xGoal", xPendCost, 1e4)
    terminal_model = crocoddyl.IntegratedActionModelEuler(
                        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel,
                                                                         terminalCostModel),
                                                                         dt)


    problem = crocoddyl.ShootingProblem(starting_condition, [runningModel] * T, terminal_model)

    fddp = crocoddyl.SolverFDDP(problem)
    fddp.th_stop = precision

    fddp.solve([], [], maxiters)



def problems(n_problems: int = 10, horizon = 100, precision = 1e-9, maxiters = 1000):
    """Solve N pendulum problems"""
    starting_conditions = np.random.randn(n_problems, 4)

    t0 = perf_counter()
    for start in starting_conditions:
        solver(starting_condition=start, T=horizon, precision=precision, maxiters=1000)
    t1 = perf_counter()
    time_taken = round((t1 - t0), 3)
    print("Done")
    f = open("pendulum.txt", "a+")
    f.write(f"\n Problems: {n_problems} , Time : {time_taken}seconds, Horizon : {horizon}, Precision : {precision}, Maxiters : {maxiters} \n ")
    f.close()


if __name__=='__main__':
    #problems(n_problems=100, horizon=100, precision=1e-9)
    #problems(n_problems=100, horizon=200, precision=1e-9)




    #problems(n_problems=100, horizon=100, precision=1e-7)
    #problems(n_problems=100, horizon=150, precision=1e-7)
    #problems(n_problems=200, horizon=200, precision=1e-7)

    problems(n_problems=1000, horizon=100, precision=1e-9)
    problems(n_problems=1000, horizon=150, precision=1e-9)