runs:int = 50, batch_size:int = 32, epochs:int = 250,
lr:float = 1e-3, n_traj:int = 100, horizon:int = 50, weights:list = [1., 1.],
precision:int = 1e-7, maxiters:int = 1000, fc1_dims:int = 20,
fc2_dims:int = 20, fc3_dims:int = 2, weight_decay:float = 5e-4):

criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.SGD(neural_net.parameters(), lr = self.lr, momentum=0.9, weight_decay=self.weight_decay, nesterov=True)


##################

 ddp.cost: 7.951580342147255 || Predicted : 7.419003997623069

 ddp.Vx[0]: [ -0.9875371  -23.67572453   8.19209894] || Net Jacobian : [  5.15564247 -15.96633868   3.22257042]

 ################

