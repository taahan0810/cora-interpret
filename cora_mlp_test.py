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

from cora_mlp_train import MyMLP, X_test, y_test, batch_size, max_epochs

testloader = DataLoader(X_test,batch_size=batch_size,shuffle=False,num_workers=0)  

model = MyMLP()
model.load_state_dict(torch.load('model.pkl'))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for epoch in range(max_epochs):
        for i,data in enumerate(testloader):

            inputs = data
            labels = data[:,-1]

            outputs = model(inputs)
            _,predictions = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            # print(f"{predictions=}")
            # print(outputs)

print(f'Accuracy of the network on the 1000 test samples: {100 * correct // total} %')

