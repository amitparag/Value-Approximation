The goal is to warmstart Crocoddyl using approximator functions to predict optimal value and policy.

Two networks will be used to warmstart crocoddyl and drive it toward optimal value function that respects HJB constraints. . 

ValueNet
............
This network will learn optimal cost. The gradient and the hessian of this network will be used in terminal model.
    1: Supervised training. Dataset size = number of trajectories * horizon: xtrain, ytrain
    2; Sobolev training,    Dataset size = number of trajectories * horizon: xtrain, ytrain1, ytrain2(=grad)
        
        The neural network has 3 hidden layers activated with tanh.
                state_space --> |->|->|-> value

        Traininig process in non-sobolev setting:
                xtrain.shape = [10000 x 3]
                ytrain.shape = [10000 x 1], where number_of_trajectories = 100 and horizon = 100

PolicyNetwork
.............
This network will predict the optimal state and control trajectories. The predictions of this network will be used as warmstart.
    1: Supervised training. Dataset size = number of trajectories.
        
        The neural network has 3 hidden layers activated with relu.
                state_space --> |->|->|-> xs, us

        Training process in supervised setting:
                xtrain.shape = [100 x 3]
                ytrain.shape = [100 x 300], 
                                          where number_of_trajectories = 100 and horizon = 50
                                          300 = 50x3 + 50x3 = xs.dimensions + us.dimensions

TODO:

1: Policy Network
2: Value Network
3: Iterative training for Value Network
    1: using gradients for optimization
    2: without using gradients for optimization

4: Training for Policy Network
    1: Check if warmstart is successful
5: Use Policy and Value Network for crocoddyl.

