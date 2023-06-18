import os
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dgl.data import CoraGraphDataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from cora-mlp-train import MyMLP

testloader = DataLoader(X_test,batch_size=batch_size,shuffle=False,num_workers=0)  
