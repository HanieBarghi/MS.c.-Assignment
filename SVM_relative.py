import scipy.io
import cvxopt
import numpy
from cvxopt import matrix
import numpy as np

train = scipy.io.loadmat('scene_train.mat')
X = train['X']
Y = train['Y'][0]

X_train = []
Y_train = []

p = 0
for i in range(Y.__len__()):
    if Y[i] == 1.0:
        p += 1

for i in range(Y.__len__()):
    if Y[i] == 1.0:
        X_train.append(X[i])
        Y_train.append(Y[i])

for i in range(Y.__len__()):
    if Y[i] == -1.0:
        X_train.append(X[i])
        Y_train.append(Y[i])

X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return numpy.array(sol['x']).reshape((P.shape[1],))


N = Y_train.__len__()
C1 = p / N
C2 = 1 - C1

P = []
for i in range(N):
    P.append([])
    for j in range(N):
        P[i].append(Y_train[i] * Y_train[j] * np.inner(np.array(X_train[i]), np.array(X_train[j])))

P = numpy.array(P)
q = -1.0 * numpy.ones(N, dtype=float)

A = matrix(Y_train.astype(float), (1, N))
b = [0.0]

t = np.ones(N) * C2
for i in range(p, N):
    t[i] = C1
tmp1 = np.diag(np.ones(N) * -1.0)
tmp2 = np.identity(N)
G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
tmp1 = np.zeros(N)
# tmp2 = np.ones(N) * C1
h = cvxopt.matrix(np.hstack((tmp1, t)))

alpha = cvxopt_solve_qp(P, q, G, h, A, b)

for i in range(alpha.__len__()):
    if alpha[i] <= 0.00000001:
        alpha[i] = 0

w = np.zeros(X_train[0].__len__())
for i in range(alpha.__len__()):
    w = w + alpha[i] * Y_train[i] * X_train[i]

for i in range(alpha.__len__()):
    if (alpha[i] > 0) and (alpha[i] < 1):
        break
w0 = 1 / Y_train[i] - np.inner(np.array(X_train[i]), w)
print(w0)

test = scipy.io.loadmat('scene_test.mat')
X_test = test['X']
Y_test = test['Y'][0]

TP1 = 0
TP2 = 0
FP1 = 0
FP2 = 0
all1 = 0
all2 = 0
for i in range(Y_test.__len__()):
    if Y_test[i] == 1:
        all1 += 1
    else:
        all2 += 1

    f = np.inner(np.array(X_test[i]), w) + w0

    if f > 0:
        if Y_test[i] > 0:
            TP1 += 1
        else:
            FP1 += 1
    else:
        if Y_test[i] < 0:
            TP2 += 1
        else:
            FP2 += 1

print('presion for G+')
print(TP1 / (TP1 + FP1))
print('presion for G-')
print(TP2 / (TP2 + FP2))

print('recall for G+')
print(TP1 / all1)
print('recall for G-')
print(TP2 / all2)
