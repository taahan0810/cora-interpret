from dgl.data import CoraGraphDataset
import torch
from cora_mlp_train import X_train, X_test, y_train, y_test, main_train, batch_size
from cora_mlp_test import main_test
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# print(y_test)

#  go from 140 to 770 with increments of 35 

mask = [False for i in range(len(y_test))]

comb_train = torch.cat([X_train, y_train],dim=1)
comb_test = torch.cat([X_test, y_test],dim=1)
masked_test = comb_test[np.logical_not(mask)]

def next_sample(mask,y_test,comb_train,comb_test):

    cnt = [0 for i in range(7)]
    for dig in range(7):
        for i in range(len(mask)):
            if y_test[i] == dig and mask[i] != True and cnt[dig] < 3:
                # print(f"{comb_train.shape=}")
                # print(f"{comb_test[i].unsqueeze(dim=0).shape=}")
                comb_train = torch.cat([comb_train,comb_test[i].unsqueeze(dim=0)])
                # comb_test = torch.cat([comb_test[:i],comb_test[i+1:]])
                mask[i] = True
                # print(f"{comb_train.shape=}")
                cnt[dig] += 1

    return comb_train, mask

test_acc_samples = []


for i in range(18):

    # training the model for a particular sample
    main_train(DataLoader(comb_train,batch_size=batch_size,shuffle=True,num_workers=0),DataLoader(masked_test,batch_size=batch_size,shuffle=True,num_workers=0),i+1)
    accuracy, csr = main_test(DataLoader(masked_test,batch_size=batch_size,shuffle=True,num_workers=0),modelname=f'model_{i+1}.pkl')
    test_acc_samples.append(accuracy)
    # train the model and spit out the test accuracy
    # print(f"{comb_train.shape=}")
    # print(f"{masked_test.shape=}")
    comb_train, mask = next_sample(mask,y_test,comb_train,comb_test)
    masked_test = comb_test[np.logical_not(mask)]

# print(f"{test_acc_samples=}")

plt.clf()

plt.style.use('fivethirtyeight')
plt.xlabel('Training Samples')
plt.ylabel('Test Accuracy')
plt.plot(np.arange(len(test_acc_samples)),test_acc_samples)
plt.savefig('sample_acc_exp.png')



                