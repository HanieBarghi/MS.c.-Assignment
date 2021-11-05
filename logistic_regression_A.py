import math
import sys
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

f = open('wifi_data.txt', 'r')
data = f.read()
f.close()
data = data.split('\n')
data_matrix = [[0 for x in range(8)] for y in range(1000)]
i = 0
for k in range(2000):
    column = data[k].split('\t')
    if int(column[7]) == 1 or int(column[7]) == 3:
        if int(column[7]) == 3:
            data_matrix[i][7] = 0
            for j in range(7):
                data_matrix[i][j] = int(column[j])
        else:
            for j in range(8):
                data_matrix[i][j] = int(column[j])
        i += 1
data_matrix = np.matrix(data_matrix)
np.random.shuffle(data_matrix)

train_data = data_matrix[:800]
x_train = train_data[:, :7]
y_train = train_data[:, 7:]
n = train_data.__len__()

test_data = data_matrix[800:1000]
x_test = test_data[:, :7]
y_test = test_data[:, 7:]

w = np.matrix('0; 0; 0; 0; 0; 0; 0')


def f(x, w):
    output = np.exp(-1 * (np.transpose(w) * x))
    return 1 / (1 + output[0, 0])


def J(w):
    output = 0
    for i in range(n):
        ff = f(np.transpose(x_train[i]), w)
        if not ff == 1:
            output = output + -1 * y_train[i, 0] * np.log(ff) - (1 - y_train[i, 0]) * np.log(1 - ff)
        else:
            output = output + -1 * y_train[i, 0] * np.log(sys.float_info.min * sys.float_info.epsilon) - (
                    1 - y_train[i, 0]) * np.log(1 - sys.float_info.min * sys.float_info.epsilon)
    return output


def gradian(w):
    output = np.matrix('0; 0; 0; 0; 0; 0; 0')
    for i in range(n):
        output = output + (f(np.transpose(x_train[i]), w) - y_train[i, 0]) * np.transpose(x_train[i])
    return output


eta = 0.00001
plot_r = []
plot_cost = []
plot_w = []
for r in range(350):
    plot_r.append(r)
    plot_cost.append(J(w))
    plot_w.append(w)
    w = w - eta * gradian(w)

plt.plot(plot_r, plot_cost)
plt.show()

result = []
for i in range(200):
    f1 = np.transpose(w) * np.transpose(x_test[i])
    f0 = 1 - f1

    if f1 > f0:
        result.append(1)
    else:
        result.append(0)

correct = 0
for i in range(200):
    if result[i] == y_test[i, 0]:
        correct += 1
print(correct/200)

matrix = [[0 for x in range(2)] for y in range(2)]
for i in range(result.__len__()):
    matrix[result[i] - 1][y_test[i,0] - 1] += 1

print(matrix)
