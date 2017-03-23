import numpy as np


def householder(X, Y):
    U = (X - Y) / np.linalg.norm(X - Y)
    I = np.identity(len(U))
    H = I - 2.0 * U.dot(U.T)
    return H

def mul_householder(X, I, J):
    H = householder(I, J)
    return H.dot(X)
        
def mul_householder_optimized(X, I, J):
    U = (I - J) / np.linalg.norm(I - J)
    S = X - 2 * U.dot(U.T.dot(X))
    return S
