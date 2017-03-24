import numpy as np
import householder as hh

#apply_householder without
def apply_householder(X, I, J, i, pos):
    n, m = X.shape
    if ( i != 0):
        Y = np.zeros((n,1))
        W = np.zeros((n,1))
        Y[i:n] = I
        W[i:n] = J
    else:
        Y = I
        W = J
        
    U = (Y - W) / np.linalg.norm(Y - W)
    if (pos == 0) : 
        S = mul_householder_left(X, U)
    else:
        S = mul_householder_right(X, U)
    print(S)
    return S

def mul_householder_right(X, U):
    S = X - 2 * U.dot(U.T.dot(X))
    return S

def mul_householder_left(X, U):
    S = X - X.dot(2 * U.dot(U.T))
    return S


def norme(X):
    n = 0
    for i in range(0, len(X), 1):
        n = n + np.square(X[i])
    return np.sqrt(n)

def bi_diagonal(A):
    BD = A
    n, m = A.shape
    Qleft = np.identity(n)
    Qright = np.identity(n)
    for i in range (0, n - 1, 1):
        X = np.copy(BD[i:n, i])
        X.shape = (n - i, 1)
        
        Y = np.zeros(n - i)
        Y[0] = norme(X)
        Y.shape = (n - i, 1)
        
        Qleft = apply_householder(Qleft, X, Y, i, 0)
        BD  = apply_householder(BD, X, Y, i, 1)
        if (i != m - 2):
            X = np.copy(BD[i, (i + 1):m])
            X.shape = (m - i - 1, 1)

            Y = np.zeros(m - i - 1)
            Y[0] = norme(X)
            Y.shape = (m - i - 1, 1)
            
            Qright = apply_householder(Qright, X, Y, i + 1, 1) 
            BD  = apply_householder(BD, X, Y, i + 1, 0)
        print("BD")
        print(np.round(BD, 2))
    return (Qleft, BD, Qright)


A = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
Qleft, BD, Qright = bi_diagonal(A)
print(A)
print(np.round(BD, 2))
print(Qleft.dot(BD).dot(Qright))
