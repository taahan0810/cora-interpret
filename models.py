import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)