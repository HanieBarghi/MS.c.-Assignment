import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x


X_train, Y_train = loadlocal_mnist(
    images_path='train-images.idx3-ubyte',
    labels_path='train-labels.idx1-ubyte')

X_train = np.array(X_train) / 255
Y_train = np.array(Y_train)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = torch.tensor(X_train, requires_grad=False, device=device).float()
target = torch.tensor(Y_train, dtype=torch.long, requires_grad=False, device=device)

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

xx = []
yy = []
for epoch in range(50):
    xx.append(epoch)
    for i in range(X.size()[0]):
        input = X[i:i + 1]
        tt = target[i:i + 1]
        optimizer.zero_grad()
        output = net(input)
        l = nn.CrossEntropyLoss()
        loss = l(output, tt)
        loss.backward()
        optimizer.step()
    yy.append(float(loss.data))
    print("Epoch {} - loss: {}".format(epoch, loss))

print(yy)
plt.plot(xx, yy)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
