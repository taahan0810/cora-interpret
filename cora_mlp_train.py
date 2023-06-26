import os
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dgl.data import CoraGraphDataset  
import wandb 

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# diff droput layers 
# remove droput from input 
# increase layers 
# 20 40 droput percentages 


class MyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MyMLP, self).__init__()

        self.layer1 = nn.Linear(input_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,output_dim)
        self.dropout1 = nn.Dropout(0.5)
        # self.dropout2 = nn.Dropout(0.5)
        # self.layer3 = nn.Linear(128,7)

    def forward(self,x):

        # print(x.shape)
        # x = torch.unsqueeze(x,dim=1)
        # print(x.shape)

        x = self.dropout1(F.relu(self.layer1(self.dropout1(x))))
        x = self.layer2(x)
        # x = self.layer3(x)

        return x
    
learning_rate = 0.01

max_epochs = 200

batch_size = 4

# Loading the dataset

dataset = CoraGraphDataset()

graph = dataset[0]
num_classes = dataset.num_classes

feat = graph.ndata['feat']

train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']

label = graph.ndata['label']

X_train, X_val, X_test = feat[train_mask], feat[val_mask], feat[test_mask]
y_train, y_val, y_test = label[train_mask], label[val_mask], label[test_mask]

# print(X_train.shape, y_train.shape)
# print(X_val.shape, y_val.shape)
# print(X_test.shape, y_test.shape)

# print(y_train.unsqueeze(dim=0).shape)
y_train = y_train.unsqueeze(dim=1)
y_val = y_val.unsqueeze(dim=1)
y_test = y_test.unsqueeze(dim=1)

trainl = DataLoader(torch.cat([X_train,y_train],dim=1),batch_size=batch_size,shuffle=True,num_workers=0)
vall = DataLoader(torch.cat([X_val,y_val],dim=1),batch_size=batch_size,shuffle=True,num_workers=0)

input_dim = feat.shape[1]
hidden_dim = 16
output_dim = dataset.num_classes

net = MyMLP(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate,weight_decay=5e-4)


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="cora_mlp_train",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "MLP",
    "dataset": "CoraGraphDataset",
    "epochs": max_epochs,
    "batch size": batch_size,
    "hidden dim": hidden_dim,
    "optimizer": "Adam"
    }
)

# TRAINING and VALIDATION

def main_train(trainloader, valloader, id=0):
    patience = 20
    best = 1e9
    best_t = 0
    cnt_wait = 0

    for epoch in range(max_epochs):

        running_loss = 0.0 
        running_vloss = 0.0 
        lab_sz = 0

        for i,data in enumerate(trainloader):

            # print(data[:,:-1].shape)

            inputs = data[:,:-1]
            labels = data[:,-1]

            # print(inputs.shape)
            # print(labels.shape)

            optimizer.zero_grad()

            outputs = net(inputs)
            # print(f"{outputs=}")
            # print(f"{labels=}")
            loss = criterion(outputs,labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            lab_sz += labels.size(0)

        print(f'[{epoch + 1}] Training loss: {running_loss/lab_sz:.3f}',end=" ")

        lab_sz = 0

        with torch.no_grad():
            for i,data in enumerate(valloader):
                
                inputs = data[:,:-1]
                labels = data[:,-1]

                # print(inputs.shape)
                # print(labels.shape)

                outputs = net(inputs)
                # print(outputs.shape)
                loss = criterion(outputs,labels.long())

                running_vloss += loss.item()
                lab_sz += labels.size(0)

        print(f'[{epoch + 1}] Validation loss: {running_vloss/lab_sz:.3f}')

        wandb.log({'Epoch':epoch+1,'Train Loss':running_loss/lab_sz,'Val Loss':running_vloss/lab_sz})

        if running_vloss/lab_sz < best:
            best = running_vloss/lab_sz
            best_t = epoch + 1
            cnt_wait = 0
            torch.save(net.state_dict(), f'model_{id}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early Stopping!')
            break

        
    print(f'Loading {best_t}th epoch')
    print("Finished Training")

    wandb.finish()

if __name__ == '__main__':
    main_train(trainloader=trainl,valloader=vall)







