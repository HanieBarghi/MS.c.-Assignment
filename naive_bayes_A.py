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

C1 = 0
C3 = 0
for y in y_train:
    if LA.norm(y) == 1:
        C1 += 1
    else:
        C3 += 1

P_C1 = C1 / 800
P_C3 = C3 / 800

# todo naive bayes for test data
# result = []
# for i in range(200):
#     probability_C1 = 1
#     probability_C3 = 1
#     for d in range(7):
#         repeat1 = 0
#         repeat3 = 0
#         for k in range(800):
#             if y_train[k, 0] == 1:
#                 if x_test[i, d] == x_train[k, d]:
#                     repeat1 += 1
#             if y_train[k, 0] == 3:
#                 if x_test[i, d] == x_train[k, d]:
#                     repeat3 += 1
#         probability_C1 = probability_C1 * repeat1 / C1
#         probability_C3 = probability_C3 * repeat3 / C3
#
#     probability_C1 = probability_C1 * P_C1
#     probability_C3 = probability_C3 * P_C3
#
#     if probability_C1 > probability_C3:
#         result.append(1)
#     else:
#         result.append(3)
#
# correct = 0
# for i in range(200):
#     if result[i] == y_test[i, 0]:
#         correct += 1
#
# print(correct/200)

# todo naive bayes for training data
result = []
for i in range(800):
    probability_C1 = 1
    probability_C3 = 1
    for d in range(7):
        repeat1 = 0
        repeat3 = 0
        for k in range(800):
            if y_train[k, 0] == 1:
                if x_train[i, d] == x_train[k, d]:
                    repeat1 += 1
            if y_train[k, 0] == 3:
                if x_train[i, d] == x_train[k, d]:
                    repeat3 += 1
        probability_C1 = probability_C1 * repeat1 / C1
        probability_C3 = probability_C3 * repeat3 / C3

    probability_C1 = probability_C1 * P_C1
    probability_C3 = probability_C3 * P_C3

    if probability_C1 > probability_C3:
        result.append(1)
    else:
        result.append(3)

correct = 0
for i in range(800):
    if result[i] == y_train[i, 0]:
        correct += 1

print(correct/800)
