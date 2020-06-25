import numpy as np
import torch
import crocoddyl
from utils import *
from feedforward_network import FeedForwardNet
from terminal_models import FeedforwardUnicycle
from tqdm import tqdm
import matplotlib.pyplot as plt

def _training( net, xtrain, ytrain,lr:float = 1e-3, batch_size:int = 128, epochs:int=1000):
    """
    @params:
        1: net = neural net to be trained
        2: xtrain, ytrain = dataset
        
    @returns:
        1: trained neural network
        
    """
    # Convert to torch dataloader
    dataset = torch.utils.data.TensorDataset(xtrain, ytrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)  
    
    net.float()
    net.train()
    
    for epoch in tqdm(range(epochs)):        
        for data, target in dataloader: 

            output = net(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    del dataset, dataloader, xtrain, ytrain
    return net



def training(runs:int=10):
    print(f"Starting {runs} Irepa runs for Feedforward Network.......\n")



    net = FeedForwardNet(fc1_dims=20, fc2_dims=20)
    print(net)
    eps = 0.
    for i in tqdm(range(runs)):
        if i == 0:
            terminal_model_unicycle = None

        else: 
            terminal_model_unicycle = FeedforwardUnicycle(net)

        init_points  = points(r=[2.-eps,1.25-eps, 0.001+eps], n = [40, 40, 20])

        states, values = statesValues(init_positions=init_points, horizon=100,
                                      terminal_model=terminal_model_unicycle,
                                      precision=1e-9)

        net = _training(net=net, 
                        xtrain=states,
                        ytrain=values)

        torch.save(net, './nets/net'+str(i)+'.pth')
        eps += 0.1
    print("Done")


if __name__=='__main__':
    training()

                





