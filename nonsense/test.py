import numpy as np
import crocoddyl


x0 = np.array([1, 2, 3])

model = crocoddyl.ActionModelUnicycle()

model.costWeights = np.array([1, 1]).T

problem = crocoddyl.ShootingProblem(x0, [model]*10, model)
ddp = crocoddyl.SolverDDP(problem)

ddp.solve()
print(len(ddp.xs))
print(len(ddp.us))