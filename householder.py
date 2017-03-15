import numpy as np


def householder(X, Y):
    U = (X - Y) / np.linalg.norm(X - Y)
    I = np.identity(len(U))
    H = I - 2.0 * U.dot(U.T)
    return H

X = np.array([[3],[4],[0]])
Y = np.array([[0],[0],[5]])
print(householder(X, Y))
