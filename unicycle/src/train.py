import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import Datagen
from value_network import ValueNet
from residual_network import ResidualNet



def train(net,
          size_of_training_dataset = 3000,
          xrange_training_data     = [-2.1, 2.1],
          yrange_training_data     = [-2.1, 2.1],
          zrange_training_data     = [-2.*np.pi, 2.*np.pi], #theta
          # DDP solver params 
          horizon                  = 30,
          precision                = 1e-9,
          maxiters                 = 1000,
          state_weight             = 1.,
          control_weight           = 1.,
          
          learning_rate            = 1e-3,
          epochs                   = 10000,
          batchsize                = 1000,
          name                     = 'value',  
          save_name                = None):
    """
    
    Initialize and train a new feedforward value network.
    
    @params:
        # Parameters of training dataset.
        1. neural_net           = network to be trained
        2: size_of_training_dataset.
        3: xrange_training_data = the range of x to sample from, when creating the training data.
        4: yrange_training_data = the range of y to sample from, when creating the training data.
        5: zrange_training_data = the range of z to sample from, when creating the training data.
            Default ranges of x, y, theta:
                x -> [-2.1, 2.1]
                y -> [-2.1, 2.1]
                z -> [-2pi, 2pi] = theta
        
        # Parameters given to crocoddyl to generate training data
        
        6: horizon        = time horizon for the ddp solver, T
        7: stop           = ddp.th_stop
        8: maxiters       = maximum iterations allowed for solver
        9: state_weight   = weight of the state vector
        10: control_weight = weight of the control vector
        
        # Parameters of the neural network
        
       
        11: learning_rate   = 1e-3
        12: epochs          = number of epochs for training
        13: batchsize       = batchsize of data during training
        14: save_name       = if a str is given, then the net will be saved. 
        
    """

    ##.......................... Training
    
    # Sample random positions for xtrain
    positions = Datagen.random_positions(size = size_of_training_dataset,
                                         xlim = xrange_training_data,
                                         ylim = yrange_training_data,
                                         zlim = zrange_training_data,
                                         as_tensor = True)
    
    # Corresponding ddp.cost for ytrain    
    values    = Datagen.values(positions = positions,
                               horizon   = horizon,
                               precision = precision,
                               maxiters  = maxiters,
                               state_weight   = state_weight,
                               control_weight = control_weight,
                               as_tensor = True)
    
    

    # Torch dataloader
    dataset = torch.utils.data.TensorDataset(positions,values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize) 



    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)  
    net.float()
    net.train()
    print(f"\n Training for {epochs} epochs... \n")
    print(f"\n Dataset size = {size_of_training_dataset}, Batchsize = {batchsize} \n")
    for epoch in tqdm(range(epochs)):        
        for data, target in dataloader: 

            outputs = net(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    ##........................... Validation
    
    # Validation dataset
    xtest     = Datagen.random_positions(size=1000, as_tensor=True)
    ytest     = Datagen.values(xtest)
    net.eval()
    ypred = net(xtest)
    error = ypred.detach() - ytest
    print(f' Mean Error:{torch.mean(error)}')
    
    if save_name is not None:
        torch.save(net, "../networks/"+save_name+".pth")
        
    else: return net
        

if __name__=='__main__':
    
    import torch
    import torch.nn as nn
    from value_network import ValueNet
    from residual_network import ResidualNet
    
        
    # Neural Network params for Value network
    nhiddenunits             = 64
    activation               = nn.Tanh()   
    net1 = ValueNet(n_hiddenUnits = nhiddenunits,        ## Initialize an untrained value network
                   activation    = activation)
    
    # Neural Network params for Residual network
    nhiddenunits             = 256
    activation               = nn.Tanh()   
    net2 = ResidualNet(n_hiddenUnits = nhiddenunits,    ## Initialize an untrained residual network
                      activation     = activation)
        

    
    
    
    train(net1,save_name='value')
    
    train(net2,save_name='residual')
