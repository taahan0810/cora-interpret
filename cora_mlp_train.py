import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import wandb

from utils import set_seeds
from models import MyMLP
from dataset import load_dataset

warnings.filterwarnings("ignore")
set_seeds()

learning_rate = 0.01

max_epochs = 200

batch_size = 4

# LOADING THE DATASET

X_train, y_train, X_val, y_val, _, _, feat, dataset = load_dataset(batch_size=batch_size)

trainl = DataLoader(torch.cat([X_train, y_train], dim=1),
                    batch_size=batch_size, shuffle=True, num_workers=0)
vall = DataLoader(torch.cat([X_val, y_val], dim=1),
                    batch_size=batch_size, shuffle=True, num_workers=0)

# HYPERPARAMETERS

input_dim = feat.shape[1]
hidden_dim = 16
output_dim = dataset.num_classes

# MODEL INTIALIZATION

net = MyMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)


# TRAINING and VALIDATION

def main_train(trainloader, valloader, id=0):

    # wandb.init(
    # # set the wandb project where this run will be logged
    #     project="cora_mlp_train",

    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": learning_rate,
    #     "architecture": "MLP",
    #     "dataset": "CoraGraphDataset",
    #     "epochs": max_epochs,
    #     "batch size": batch_size,
    #     "hidden dim": hidden_dim,
    #     "optimizer": "Adam"
    #     }
    # )
    patience = 20
    best = 1e9
    best_t = 0
    cnt_wait = 0

    for epoch in range(max_epochs):

        running_loss = 0.0
        running_vloss = 0.0
        lab_sz = 0

        for i, data in enumerate(trainloader):

            inputs = data[:, :-1]
            labels = data[:, -1]

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            lab_sz += labels.size(0)

        print(
            f'[{epoch + 1}] Training loss: {running_loss/lab_sz:.3f}',
            end=" ")

        lab_sz = 0

        with torch.no_grad():
            for i, data in enumerate(valloader):

                inputs = data[:, :-1]
                labels = data[:, -1]

                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                running_vloss += loss.item()
                lab_sz += labels.size(0)

        print(f'[{epoch + 1}] Validation loss: {running_vloss/lab_sz:.3f}')

        # wandb.log({'Epoch':epoch+1,'Train Loss':running_loss/lab_sz,'Val Loss':running_vloss/lab_sz})

        if running_vloss / lab_sz < best:
            best = running_vloss / lab_sz
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

    # wandb.finish()


if __name__ == '__main__':
    main_train(trainloader=trainl, valloader=vall)
