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

points1 = Datagen.uniformSphere(5)
points2 = Datagen.uniformSphere(5)

a = torch.rand(2, 3)
b = torch.norm(a, 'fro', dim=1)
print(b)