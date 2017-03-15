import numpy as np


def householder(X, Y):
    U = (X - Y) / np.linalg.norm(X - Y)
    I = np.identity(len(U))
    H = I - 2.0 * U.dot(U.T)
    return H

def mul_householder(X, H):
    return H.dot(X)
        
def mul_householder_optimized(X, I, J):
    I = np.array([[3], [4], [0]])
    J = np.array([[0], [0], [5]])
    U = (I - J) / np.linalg.norm(I - J)
    S = X - 2 * U.dot(U.T.dot(X))
    return S

I = np.array([[3], [4], [0]])
J = np.array([[0], [0], [5]])
X = np.array([[3, 4, 7], [4, 7, 9], [0, 0, 0]])
print(mul_householder(X, householder(I, J)))
print(mul_householder_optimized(X, I, J))
