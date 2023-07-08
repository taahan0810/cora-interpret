from dgl.data import CoraGraphDataset
import torch
from cora_mlp_train import X_train, X_test, y_train, y_test, X_val, y_val, main_train, batch_size
from cora_mlp_test import main_test
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from cora_gcn_train import data,dataset,main_gcn

seed_value = 123
np.random.seed(seed_value)
torch.manual_seed(seed_value)


# print(f"{X_test=}")
indices = torch.Tensor([i for i in range(len(y_test))])

# print(f"{indices.shape=}")


new_X_val,new_X_test,new_y_val,new_y_test = train_test_split(X_test,torch.cat([y_test,indices.unsqueeze(dim=1)],dim=1), test_size=0.50, random_state=seed_value, stratify=y_test)

# print(f"{new_y_test=}") 
gcn_test_mask = [False for i in range(X_test.shape[0])]

# print(f"{new_y_test[0][1].item()=}")

for i in range(new_y_test.shape[0]):
    gcn_test_mask[int(new_y_test[i][1].item())] = True

gcn_val_mask = [ not v for v in gcn_test_mask ]

custom_val_mask = [ True for i in range(len(data.val_mask)) ]
custom_test_mask = [ True for i in range(len(data.test_mask)) ]

for i in range(len(data.test_mask)):
    if custom_val_mask[i] & data.test_mask[i]:
        custom_val_mask[i] = custom_val_mask[i] & data.test_mask[i] & gcn_val_mask[i-1708]

for i in range(len(data.test_mask)):
    if custom_test_mask[i] & data.test_mask[i]:
        custom_test_mask[i] = custom_test_mask[i] & data.test_mask[i] & gcn_test_mask[i-1708]


new_y_val = new_y_val[:,0]
new_y_test = new_y_test[:,0]

# print(new_train_mask)

# print(f"{X_train.shape=}")
# print(f"{X_val.shape=}")
new_X_train = torch.cat([X_train, X_val],dim=0)
new_y_train = torch.cat([y_train, y_val],dim=0)

mask = [False for i in range(len(data.train_mask))]

# print(f"{new_X_train.shape=}")
# print(f"{new_y_train.shape=}")


def next_sample(mask,comb_train):

    cnt = [0 for i in range(7)]
    for dig in range(7):
        for i in range(len(comb_train)):
            if comb_train[:,-1][i] == dig and mask[i] != True and cnt[dig] < 5:
                mask[i] = True
                # print(f"{comb_train.shape=}")
                cnt[dig] += 1

    return mask

# print(f"{new_y_val.shape=}")

comb_train = torch.cat([new_X_train, new_y_train],dim=1)
comb_val = torch.cat([new_X_val, new_y_val.unsqueeze(dim=1)],dim=1)
comb_test = torch.cat([new_X_test, new_y_test.unsqueeze(dim=1)],dim=1)

test_acc_samples = []
samples = []
test_acc_gcn_samples = []

# for i in range(len(data.val_mask)):
#     if data.val_mask[i]:
#         print(f"{i=}")

if __name__ == "__main__":

    for i in range(18):

        mask = next_sample(mask,comb_train)
        print(f"{comb_train[mask[:640]].shape=}")

        main_train(DataLoader(comb_train[mask[:640]],batch_size=batch_size,shuffle=True,num_workers=0),DataLoader(comb_val,batch_size=batch_size,shuffle=True,num_workers=0),i+1)
        accuracy, csr = main_test(DataLoader(comb_test,batch_size=batch_size,shuffle=False,num_workers=0), modelname=f'model_{i+1}.pkl')

        gcn_acc = main_gcn(dataset,data,mask,custom_val_mask,custom_test_mask)

        print(f'Test Accuracy: {gcn_acc:.4f}')

        samples.append(comb_train[mask[:640]].shape[0])
        test_acc_gcn_samples.append(100 * gcn_acc)
        test_acc_samples.append(accuracy)

    plt.clf()

    plt.style.use('fivethirtyeight')
    plt.xlabel('Training Samples')
    plt.ylabel('Test Accuracy')
    plt.plot(samples,test_acc_samples)
    plt.plot(samples,test_acc_gcn_samples)
    plt.savefig('sample_acc_exp.png')


