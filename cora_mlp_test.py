import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from models import MyMLP
from utils import set_seeds
from dataset import load_dataset

batch_size = 4

set_seeds()
_, _, _, _, X_test, y_test, _, _ = load_dataset(batch_size=batch_size)

testl = DataLoader(torch.cat([X_test,y_test],dim=1),batch_size=batch_size,shuffle=True,num_workers=0)  

def main_test(testloader,modelname='model_0.pkl'):
    # remove the hardcoded values afterwards
    model = MyMLP(1433,16,7)
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
    sns.heatmap(cf_matrix, annot=True, fmt='g',cmap='Blues')
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


