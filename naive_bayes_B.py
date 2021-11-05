import math
import sys

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

f = open('wifi_data.txt', 'r')
data = f.read()
f.close()
data = data.split('\n')
data_matrix = [[0 for x in range(8)] for y in range(2000)]
i = 0
for k in range(data_matrix.__len__()):
    column = data[k].split('\t')
    for j in range(8):
        data_matrix[i][j] = int(column[j])
    i += 1
data_matrix = np.matrix(data_matrix)
np.random.shuffle(data_matrix)

train_data = data_matrix[:1600]
x_train = train_data[:, :7]
y_train = train_data[:, 7:]
n = train_data.__len__()

test_data = data_matrix[1600:2000]
x_test = test_data[:, :7]
y_test = test_data[:, 7:]

C1 = 0
C2 = 0
C3 = 0
C4 = 0
for y in y_train:
    if LA.norm(y) == 1:
        C1 += 1
    elif LA.norm(y) == 2:
        C2 += 1
    elif LA.norm(y) == 3:
        C3 += 1
    else:
        C4 += 1

P_C1 = C1 / 2000
P_C2 = C2 / 2000
P_C3 = C3 / 2000
P_C4 = C4 / 2000

# todo naive bayes for test data
# result = []
# for i in range(400):
#     probability_C1 = 1
#     probability_C2 = 1
#     probability_C3 = 1
#     probability_C4 = 1
#     for d in range(7):
#         repeat1 = 0
#         repeat2 = 0
#         repeat3 = 0
#         repeat4 = 0
#         for k in range(1600):
#             if y_train[k, 0] == 1:
#                 if x_test[i, d] == x_train[k, d]:
#                     repeat1 += 1
#             elif y_train[k, 0] == 2:
#                 if x_test[i, d] == x_train[k, d]:
#                     repeat2 += 2
#             elif y_train[k, 0] == 3:
#                 if x_test[i, d] == x_train[k, d]:
#                     repeat3 += 3
#             else:
#                 if x_test[i, d] == x_train[k, d]:
#                     repeat4 += 4
#         probability_C1 = probability_C1 * repeat1 / C1
#         probability_C2 = probability_C2 * repeat2 / C2
#         probability_C3 = probability_C3 * repeat3 / C3
#         probability_C4 = probability_C4 * repeat4 / C4
#
#     probability = []
#     probability.append(probability_C1 * P_C1)
#     probability.append(probability_C2 * P_C2)
#     probability.append(probability_C3 * P_C3)
#     probability.append(probability_C4 * P_C4)
#
#     result.append(probability.index(max(probability)) + 1)
#
# correct = 0
# for i in range(400):
#     if result[i] == y_test[i, 0]:
#         correct += 1
#
# print(correct / 400)

# todo naive bayes for training data
result = []
for i in range(1600):
    print(i)
    probability_C1 = 1
    probability_C2 = 1
    probability_C3 = 1
    probability_C4 = 1
    for d in range(7):
        repeat1 = 0
        repeat2 = 0
        repeat3 = 0
        repeat4 = 0
        for k in range(1600):
            if y_train[k, 0] == 1:
                if x_train[i, d] == x_train[k, d]:
                    repeat1 += 1
            elif y_train[k, 0] == 2:
                if x_train[i, d] == x_train[k, d]:
                    repeat2 += 2
            elif y_train[k, 0] == 3:
                if x_train[i, d] == x_train[k, d]:
                    repeat3 += 3
            else:
                if x_train[i, d] == x_train[k, d]:
                    repeat4 += 4
        probability_C1 = probability_C1 * repeat1 / C1
        probability_C2 = probability_C2 * repeat2 / C2
        probability_C3 = probability_C3 * repeat3 / C3
        probability_C4 = probability_C4 * repeat4 / C4

    probability = []
    probability.append(probability_C1 * P_C1)
    probability.append(probability_C2 * P_C2)
    probability.append(probability_C3 * P_C3)
    probability.append(probability_C4 * P_C4)

    result.append(probability.index(max(probability)) + 1)

matrix = [[0 for x in range(4)] for y in range(4)]
for i in range(result.__len__()):
    matrix[result[i] - 1][y_train[i,0] - 1] += 1

print(matrix)
