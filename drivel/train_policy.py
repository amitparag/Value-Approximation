import numpy as np
import crocoddyl
import torch
import torch.nn as nn
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from policy_network import PolicyNetwork
from helper import Datagen
from helper import Solver

 
torch.set_default_dtype(torch.double)
torch.cuda.empty_cache()

#...... Hyperparameters

BATCHSIZE        = 32
TRAJCECTORIES    = 1000
lr               = 1e-3
EPOCHS           = 3000
HORIZON          = 100
DECAY            = 5e-6
PRECISION        = 1e-9
MAXITERS         = 1000
DEVICE           = 'cuda'




#....... Initlialize an empty net

neural_net = PolicyNetwork(state_dims=3, horizon=HORIZON+1, policy_dims=3, fc1_dims=100, fc2_dims=100,fc3_dims=100, activation=nn.ReLU6(), device = DEVICE)



#......... Training Data Generation

starting_points         = Datagen.griddedData(n_points=TRAJCECTORIES)
x_train, y_train         = [], []
for starting_point in starting_points:
    model               = crocoddyl.ActionModelUnicycle()
    model.costWeights   = np.array([1.,1.]).T
    problem             = crocoddyl.ShootingProblem(starting_point.T, [model]*HORIZON, model)
    ddp                 = crocoddyl.SolverDDP(problem)
    ddp.th_stop         = PRECISION
    ddp.solve([], [], MAXITERS)
    xs                  = np.array(ddp.xs)
    xs                  = xs[:,:]
    xs                  = xs.flatten().tolist()
    x_train.append(starting_point)
    y_train.append(xs)

x_train = np.array(x_train)
y_train = np.array(y_train)


x_test  = x_train[0:100,:]
y_test  = y_train[0:100,:]

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

x_test = torch.Tensor(x_test).to(DEVICE)
y_test = torch.Tensor(y_test).to(DEVICE)


# Convert to torch dataloader
dataset = torch.utils.data.TensorDataset(x_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BATCHSIZE, shuffle=True)

#......  CRITERIA
criterion1 = torch.nn.MSELoss(reduction='sum')
criterion2 = torch.nn.L1Loss(reduction='mean')


#.......  OPTIMIZER
optimizer = torch.optim.ASGD(neural_net.parameters(), lr = lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay= DECAY)
#optimizer = torch.optim.SGD(neural_net.parameters(), lr = lr, momentum=0.9, weight_decay=DECAY, nesterov=True)
#optimizer = torch.optim.Adam(neural_net.parameters(), lr = lr, betas=[0.5, 0.9], weight_decay=DECAY)

for epoch in range(EPOCHS):
    for data, target in dataloader:
        neural_net.train()
        optimizer.zero_grad()

        data        = data.to(DEVICE)
        target      = target.to(DEVICE)
        
        output      = neural_net(data)
        loss        = torch.sqrt(criterion1(output, target)) + criterion1(output, target)
        
        loss.backward()
        optimizer.step()

    prediction  = neural_net(x_test)
    mse         = torch.mean((prediction-y_test)**2)
    mae         = torch.mean(torch.abs(prediction-y_test))
    print(f"Epoch {epoch+1} :: mse = {mse}, mae = {mae}")

#......... Sanity Check
x  = [np.random.uniform(-1.5,1.5), np.random.uniform(-1.5,1.5), np.random.uniform(-np.pi/4, np.pi/4)]
np.round_(x, 4)

x1 = torch.Tensor(x).to(DEVICE)
xs1 = neural_net.guessAPolicy(x1)

x0 = np.array(x)
model               = crocoddyl.ActionModelUnicycle()
model.costWeights   = np.array([1.,1.]).T
problem             = crocoddyl.ShootingProblem(x0.T, [model]*20, model)
ddp                 = crocoddyl.SolverDDP(problem)
ddp.solve([], [], MAXITERS)
xs2 = np.array(ddp.xs)
print(ddp.iter)


plt.plot(xs1[0:20,0], xs1[0:20,1], '--*', label = "Warmstart Guess")
plt.plot(xs2[:,0], xs2[:,1], '--*', label = 'Crocoddyl')
plt.legend()
plt.savefig("warmstart2.png")
plt.show()



model               = crocoddyl.ActionModelUnicycle()
model.costWeights   = np.array([1.,1.]).T
problem             = crocoddyl.ShootingProblem(x0.T, [model]*HORIZON, model)
ddp                 = crocoddyl.SolverDDP(problem)
ddp.solve(xs1[0:20, :], [], MAXITERS)
print(ddp.iter)

