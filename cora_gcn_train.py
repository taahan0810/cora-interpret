import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

import numpy as np

seed_value = 123
np.random.seed(seed_value)
torch.manual_seed(seed_value)


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


# Set the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0].to(device)

# Initialize the GCN model
input_dim = dataset.num_features
hidden_dim = 16
output_dim = dataset.num_classes
model = GCN(input_dim, hidden_dim, output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Train the model
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        _, predicted = output.max(dim=1)
        correct = predicted[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

# Evaluate the model on the test set
model.eval()
output = model(data.x, data.edge_index)
_, predicted = torch.max(output,1)
correct = predicted[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print(f'Test Accuracy: {acc:.4f}')
