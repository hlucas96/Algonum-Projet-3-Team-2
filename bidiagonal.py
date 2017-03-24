import numpy as np
import householder as hh

def apply_householder(X, I, J, i):
    n, m = X.shape
    U_t = (I - J) / np.linalg.norm(I - J)
    U = implant(n, m, U_t, i)
    S = X - 2 * U.dot(U.T.dot(X))
    return S

def implant(n, m, Q, i):
    I = np.identity(n)
    for k in range (i, n, 1):
        for l in range (i, m, 1):
            I[k][l] = Q[k - i][l - i]
    return I

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
        
        #Q1_t = hh.householder(X, Y)
        #Q1 = implant(n, m, Q1_t, i)
        #Qleft = Qleft.dot(Q1)
        BD  = apply_householder(BD, X, Y, i)
        if (i != m - 2):
            X = np.copy(BD[i, (i + 1):m])
            X.shape = (m - i - 1, 1)

            Y = np.zeros(m - i - 1)
            Y[0] = norme(X)
            Y.shape = (m - i - 1, 1)
            
            #Q2_t = hh.householder(X, Y)
            #Q2 = implant(n, m, Q2_t, i + 1)
            #Qright = Qright.dot(Q2)
            
            BD  = apply_householder(BD.T, X, Y, i + 1)
    return (Qleft, BD, Qright)


A = np.array([[1, 2, 3], [4, 5, 9], [7, 8, 9]])
Qleft, BD, Qright = bi_diagonal(A)
print(A)
print(np.round(BD, 2))
print(Qleft.dot(BD).dot(Qright))
