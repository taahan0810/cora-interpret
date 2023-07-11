from dgl.data import CoraGraphDataset
import torch
from torch.utils.data import DataLoader

def load_dataset(batch_size):
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

    y_train = y_train.unsqueeze(dim=1)
    y_val = y_val.unsqueeze(dim=1)
    y_test = y_test.unsqueeze(dim=1)

    return X_train, y_train, X_val, y_val, X_test, y_test, feat, dataset