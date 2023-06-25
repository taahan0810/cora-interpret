import os
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dgl.data import CoraGraphDataset
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from cora_mlp_train import MyMLP, X_test, y_test, batch_size

# print(f"{X_test.shape}")
# print(f"{y_test.shape}")

testl = DataLoader(torch.cat([X_test,y_test],dim=1),batch_size=batch_size,shuffle=True,num_workers=0)  

def main_test(testloader,modelname='model_0.pkl'):
    model = MyMLP()
    model.load_state_dict(torch.load(modelname))
    model.eval()

    correct = 0
    total = 0

    y_pred = []
    shuff_y_test = []

    with torch.no_grad():
        for i,data in enumerate(testloader):

            inputs = data[:,:-1]
            labels = data[:,-1]

            outputs = model(inputs)
            _,predictions = torch.max(outputs, 1)

            # print(predictions.tolist())

            y_pred.extend(predictions.tolist())
            shuff_y_test.extend(labels.tolist())
            # total += labels.size(0)
            # correct += (predictions == labels).sum().item()
            # print(f"{predictions=}")
            # print(outputs)

    plt.clf()

    cf_matrix = confusion_matrix(shuff_y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g')
    plt.savefig('cf_matrix.png')

    accuracy = 100 * accuracy_score(shuff_y_test,y_pred)
    csr = classification_report(shuff_y_test, y_pred)

    return accuracy, csr
    # print(f'Precision : {100 * precision_score(shuff_y_test,y_pred)} %')
    # print(f'Recall : {100 * recall_score(shuff_y_test,y_pred)} %')
    # print(f'F1 Score : {100 * f1_score(shuff_y_test,y_pred)} %')

if __name__ == '__main__':
    accuracy, csr = main_test(testloader=testl)

    print(f'Accuracy : {accuracy} %')
    print(csr)


