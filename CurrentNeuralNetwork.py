import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from six.moves import cPickle as pickle
import numpy as np
import pickle
import torchvision
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # first hidden layer
        self.cl1 = nn.Conv2d(3, 64, 5, stride=1, padding=0)
        # second hidden layer
        self.cl2 = nn.Conv2d(64, 64, 5, stride=1, padding=0)
        self.pl2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # third hidden layer
        self.cl3 = nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.pl3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # forth hidden layer
        self.ll4 = nn.Linear(1600, 256)
        # output layer
        self.ll5 = nn.Linear(256, 10)

    def forward(self, x):
        # activation function of first layer
        x1 = F.relu(self.cl1(x))
        # activation function of second layer
        x2 = F.relu(self.cl2(x1))
        x2 = self.pl2(x2)
        # activation function of third layer
        x3 = F.relu(self.cl3(x2))
        x3 = self.pl3(x3)

        temp = x3.view(-1, 1600)

        # activation function of forth layer
        x4 = F.relu(self.ll4(temp))
        # activation function of output layer
        x5 = self.ll5(x4)

        return x1, x2, x3, x4, x5


def load_cifar100_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root='./cifardata', train=True, download=True, transform=transform)

    test_data = torchvision.datasets.CIFAR100(root='./cifardata', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                  shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                shuffle=False, num_workers=2)

    return train_loader, testloader


# loading the data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
new_train_data, new_test_data = load_cifar100_data()
# class of bicycle, mountain, bicycle, table, dolphin
classes = [8, 13, 30, 49, 84]

new_cnn = CNN()
new_cnn.load_state_dict(torch.load('./cifardata/my_cnn'))
new_cnn.eval()

for item in new_cnn.parameters():
    item.requires_grad = False

new_cnn.ll4 = nn.Linear(1600, 256)
new_cnn.ll5 = nn.Linear(256, 5)

cnn = CNN()
cnn.to(device)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# train the system
xx = []
y1 = []
y2 = []
y3 = []
for epoch in range(50):
    num = 0
    mean_loss = 0
    print(epoch)
    xx.append(epoch)
    for i, data in enumerate(new_train_data, 0):
        inp, target = data[0].to(device), data[1].to(device)
        if target in classes:
            num += 1
            optimizer.zero_grad()
            output = cnn(inp)
            loss = criterion(output[4], target)
            loss.backward()
            optimizer.step()
            mean_loss += float(loss.data)
            outputs = cnn(inp)

    # loss function
    y1.append(mean_loss / num)

    # accuracy for train data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in new_train_data:
            inp, labels = data[0].to(device), data[1].to(device)
            outputs = cnn(inp)
            _, predicted = torch.max(outputs[4].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    y2.append(100 * correct / total)

    # accuracy for train data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in new_test_data:
            inp, labels = data[0].to(device), data[1].to(device)
            outputs = cnn(inp)
            _, predicted = torch.max(outputs[4].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    y3.append(100 * correct / total)

    print(str(y1[epoch]) + '\t' + str(y2[epoch]) + '\t' + str(y3[epoch]))

# show loss function per epoch
print(y1)
plt.plot(xx, y1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# show accuracy per epoch
print(y2)
plt.plot(xx, y2)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# show accuracy per epoch
print(y3)
plt.plot(xx, y3)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()