import csv
import numpy as np
import math
from numpy import linalg as LA
import statistics
import matplotlib.pyplot as plt


def inputs():
    X = []
    Y = []
    with open('camera_dataset.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        i = -1
        for row in csv_reader:
            if '' in row:
                continue
            else:
                if i >= 0:
                    X.append([])
                    X[i].append(1.0)
                    for r in row[1:12]:
                        X[i].append(float(r))
                    Y.append(float(row[12]))
                i += 1
    for j in range(1, 12):
        m = [float(item[j]) for item in X]
        m = max(m)
        for i in range(X.__len__()):
            X[i][j] = X[i][j] / m
    m = max(Y)
    for i in range(Y.__len__()):
        Y[i] = Y[i] / m
    return X, Y


def loss(Y_train, X_train):
    return LA.norm(Y_train - np.matmul(X_train, W), 2)


def train():
    ys = []
    xs = []
    alpha = math.pow(10, -5)
    W = [0 for i in range(12)]
    i = 0
    while True:
        xs.append(i)
        W = W + alpha * np.matmul(np.transpose(X_train),
                                  (Y_train - np.matmul(X_train, W)))
        ys.append(loss(Y_train, X_train, W))
        if i >= 3:
            if math.fabs(ys[i] - ys[i - 1]) < math.pow(10, -5):
                break
        i += 1
    plt.plot(xs, ys)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.show()
    return W


def inference(x):
    return np.matmul(np.transpose(W), x)


def evaluate():
    x = np.ones((12,))
    for i in range(12):
        a = [float(item[i]) for item in X_train]
        x[i] = statistics.mean(a)
    x[1] = 2019
    xs = []
    ys = []
    res = 1024
    print(W[2])
    for i in range(100):
        xs.append(res)
        x[2] = res
        ys.append(inference(x))
        res += 10.24

    plt.plot(xs, ys)
    plt.xlabel('max resolution')
    plt.ylabel('price')
    plt.show()


X_train, Y_train = inputs()
X_train = np.array(X_train)
Y_train = np.array(Y_train)
W = train()
print(W)
evaluate()
