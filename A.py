import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
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
        x = F.relu(self.cl1(x))
        # activation function of second layer
        x = F.relu(self.cl2(x))
        x = self.pl2(x)
        # activation function of third layer
        x = F.relu(self.cl3(x))
        x = self.pl3(x)

        x = x.view(-1, 1600)

        # activation function of forth layer
        x = F.relu(self.ll4(x))
        # activation function of output layer
        x = self.ll5(x)

        return x


def load_cifar10_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./cifardata', train=True,
                                            download=True, transform=transform)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=1,
                                             shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./cifardata', train=False,
                                           download=True, transform=transform)
    test_data = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)

    return train_data, test_data


# loading the data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
train_data, test_data = load_cifar10_data()

# set the device
# cnn = CNN()
# cnn.to(device)
# optimizer = optim.Adam(cnn.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()
#
# # train the system
# xx = []
# y1 = []
# y2 = []
# y3 = []
# for epoch in range(100):
#     mean_loss = 0
#     print(epoch)
#     xx.append(epoch)
#     for i, data in enumerate(train_data, 0):
#         inp, target = data[0].to(device), data[1].to(device)
#         optimizer.zero_grad()
#         output = cnn(inp)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         mean_loss += float(loss.data)
#         outputs = cnn(inp)
#
#     # loss function
#     y1.append(mean_loss / 50000)
#
#     # accuracy for train data
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in train_data:
#             inp, labels = data[0].to(device), data[1].to(device)
#             outputs = cnn(inp)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     y2.append(100 * correct / total)
#
#     # accuracy for train data
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_data:
#             inp, labels = data[0].to(device), data[1].to(device)
#             outputs = cnn(inp)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     y3.append(100 * correct / total)
#
#     print(str(y1[epoch]) + '\t' + str(y2[epoch]) + '\t' + str(y3[epoch]))
#
# # show loss function per epoch
# print(y1)
# plt.plot(xx, y1)
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()
#
# # show accuracy per epoch
# print(y2)
# plt.plot(xx, y2)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
#
# # show accuracy per epoch
# print(y3)
# plt.plot(xx, y3)
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()


# part B