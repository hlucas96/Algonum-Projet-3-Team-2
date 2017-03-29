import numpy as np

#return a householder matrix from two column vectors.
def householder(X, Y):
    U = (X - Y) / np.linalg.norm(X - Y)
    I = np.identity(len(U))
    H = I - 2.0 * U.dot(U.T)
    return H

#return the the scalar product between householder(I, J) and X.
def mul_householder(X, I, J):
    H = householder(I, J)
    return H.dot(X)

#same function but optimized: householder not calculated, just apply the transformation to the X vector step by step.
def mul_householder_optimized(X, I, J):
    U = (I - J) / np.linalg.norm(I - J)
    S = X - 2 * U.dot(U.T.dot(X))
    return S


#calculate the norme of the vector X
def norme(X):
    n = 0
    for i in range(0, len(X), 1):
        n = n + np.square(X[i])
    return np.sqrt(n)
