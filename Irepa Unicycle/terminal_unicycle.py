import torch
import numpy as np
import crocoddyl
from feedforward_network import FeedForwardNet
from residual_network import ResidualNet

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
        
        
class ResidualUnicycle(crocoddyl.ActionModelAbstract):
    """
    This includes the residual network in crocoddyl. An additional param has to added in init
    to enable gauus approximation
    
    """
    def __init__(self, neural_net, use_gauss_approx=False):
        """
        The neural network is a residual network:
            x ---> R**2, where R is the residual matrix.
            
        @params:
            1: neural_net        = must be the residual network
            2: use_gauss_approx  = to enable gauss approximation of the gradient and hessian
            
        A. The gradient(& hessian) needed for data.Lx can be calculated in two ways:
            
            1:   Calculate the Jacobian of the output of the neural network w.r.t input to get Lx
            
            1.1: Calculate the Hessian of the output of the neural network w.r.t input to get Lxx
            
        B. Using Gauss approximation, the gradient and hessians are calculated like so:
            1:  Gradient = data.Lx = J.T @ r
                                                                          
            1.1 Hessian = data.Lxx = J.T @ J,
                                       where J is the jacobian of the residual matrix and r is the 
                                       residual matrix
                                       
        if A is used, then the activation of the residual network must be tanh()
        """
        
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.net = neural_net
        self.use_gauss = use_gauss_approx

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
            
        x = torch.tensor(x, dtype = torch.float32).resize_(1, 3)
        
        if self.use_gauss:
            data.Lx = self.net.gradient(x).detach().numpy()
            data.Lxx = self.net.newton_hessian(x).detach().numpy()
            
        else:
            data.Lx = self.net.jacobian(x).detach().numpy()
            data.Lxx = self.net.hessian(x).detach().numpy() 

        
