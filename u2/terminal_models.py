import torch
import numpy as np
import crocoddyl
from feedforward_network import FeedForwardNet






class FeedforwardUnicycle(crocoddyl.ActionModelAbstract):
    """
    This includes a feedforward network in crocoddyl
    
    """
    def __init__(self, neural_net):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = neural_net

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
            
        x = torch.tensor(x, dtype = torch.float32).resize_(1, 3)
        
        # Get the cost
        with torch.no_grad():
            data.cost = self.net(x).item()


    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
            
        # This is irritating. Converting numpy to torch everytime.
        x = torch.tensor(x, dtype = torch.float32).resize_(1, 3)
        
        data.Lx = self.net.jacobian(x).detach().numpy()
        data.Lxx = self.net.hessian(x).detach().numpy()




