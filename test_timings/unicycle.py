import crocoddyl
import numpy as np
from time import perf_counter
def solver(starting_condition, T = 30, precision = 1e-9, maxiters = 1000):
    """Solve one crocoddyl problem"""
    
    model = crocoddyl.ActionModelUnicycle()
    model.costWeights = np.array([1., 1.]).T

    problem = crocoddyl.ShootingProblem(starting_condition, [model] * T, model)
    ddp = crocoddyl.SolverDDP(problem)
    ddp.th_stop = precision
    ddp.solve([], [], maxiters)
    


def trajectories(n_problems:int = 10000, horizon = 30, precision = 1e-9, maxiters = 1000):
    """Solve n problems with unicycle"""

    starting_conditions = np.random.randn(n_problems, 3)

    t0 = perf_counter()
    for start in starting_conditions:
        solver(starting_condition=start, T=horizon, precision=precision, maxiters=1000)
    t1 = perf_counter()
    time_taken = round((t1 - t0), 3)
    print("Done")
    f = open("unicycle.txt", "a+")
    f.write(f"\n Problems: {n_problems} , Time : {time_taken}seconds, Horizon : {horizon}, Precision : {precision}, Maxiters : {maxiters} \n ")
    f.close()
    

if __name__=='__main__':
    trajectories(n_problems=10000, horizon=30, precision=1e-9)
    trajectories(n_problems=10000, horizon=100, precision=1e-9)
    trajectories(n_problems=10000, horizon=200, precision=1e-9)

    trajectories(n_problems=10000, horizon=30, precision=1e-7)
    trajectories(n_problems=10000, horizon=100, precision=1e-7)
    trajectories(n_problems=10000, horizon=200, precision=1e-7)


