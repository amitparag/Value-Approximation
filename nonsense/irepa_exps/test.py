import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import Solver
from utils import Datagen
from utils import TerminalModelUnicycle
from network import ValueNet
import torch
import random

points = Datagen.uniformSphere(10)

a, b, d = Solver(initial_configs=points).solveNodesValuesGrads(False)

d = torch.rand(1000, 1)

e = d[0:10]

print(e.dim())


#print(type(a))
#a = torch.Tensor(a)
#print(type(a))
#print(a)