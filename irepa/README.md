    Goal: Generate trajectories and use subtrajectories for supervised learning of value and policy.

    Using subtrajectories for supervised learning of policy requires the training data to be processed, since the subtrajectories generated are of variable length.
    This problem can be solved in three ways :
        1: Use the obvious padding approach common in RNNs where sequences of data, if of unequal length, are padded with zeros, to make them equal.
        
        2: Use another representation for the state and control trajectories which removes variability in the size of the dataset. 
           Learn this representation and use it to generate the policy.
        
        3: Exploit the markovian structure of the dataset. The dataset is essentially policy at ith step, policy at jth step (given the policy at ith step) and so on.
           
           Therefore, two neural nets(or even one) can be trained to obtain subsequent state and control, given the current state. 
           This way the neural net when queried for any current state will immediately predict the next state. This immediately shows that the neural network can be recursively queried to generate the entire policy for the horizon.


     
    Method:
    
    Two approximators are used:
        1: Value Approximator                  : to predict value 
        2: State Trajectory Approximator       : to predict state trajectory

    
    #..... HYPERPARAMS

    HORIZON              = 30    
    WEIGHTS              = [1, 1] -->  state and control weights
    NTRAJ                = 500


    Repeat until Convergence:
        1: Randomly sample NTRAJ starting points
        2: With crocoddyl(no terminal network inside crocoddyl, if irepa_iteration = 1), generate trajectories using the starting points
        3: Generate two datasets using those trajectories
        
            3.1  Dataset 1 : For Value Approximator
                        Size of dataset: 
                            xtrain = (NTRAJ*HORIZON, 3)
                            ytrain = (NTRAJ*HORIZON, 1)

                        

            3.2  Dataset 2: For State Trajectory Approximator
                        Size of dataset:
                            xtrain = (NTRAJ*HORIZON, 3)
                            ytrain = (NTRAJ*HORIZON, 3)
        
        4: Train all the Approximators
        5: If convergence criteria is met, exit.

    Convergence Criteria:
        1: Convergence criteria will be established if stopping criteria is a constant close to zero.
                    
