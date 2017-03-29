import numpy as np
import householder as hh

#apply_householder to the X matrix 
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
    return S

#apply scalar product between householder(I, J) and X
def mul_householder_right(X, U):
    S = X - 2 * U.dot(U.T.dot(X))
    return S

#apply scalar product between X and householder(I, J)
def mul_householder_left(X, U):
    S = X - X.dot(2 * U.dot(U.T))
    return S

#divide the A matrix into: Qleft, BD(bidiagonal matrix), Qright with: Qleft * BD * Qright = A
#debug is a flag to know if we are in debug mode or not
def bi_diagonal(A, Debug):
    BD = A
    n, m = A.shape
    Qleft = np.identity(n)
    Qright = np.identity(n)
    for i in range (0, n - 1, 1):
        #put the i column to zero
        X = np.copy(BD[i:n, i])
        X.shape = (n - i, 1)
        
        Y = np.zeros(n - i)
        Y[0] = hh.norme(X)
        Y.shape = (n - i, 1)
        
        Qleft = apply_householder(Qleft, X, Y, i, 0)
        BD  = apply_householder(BD, X, Y, i, 1)
        if (i != m - 2):
            #put the i line to zero
            X = np.copy(BD[i, (i + 1):m])
            X.shape = (m - i - 1, 1)

            Y = np.zeros(m - i - 1)
            Y[0] = hh.norme(X)
            Y.shape = (m - i - 1, 1)
            
            Qright = apply_householder(Qright, X, Y, i + 1, 1) 
            BD  = apply_householder(BD, X, Y, i + 1, 0)
        if(Debug == 1):
            #test if the invariant is still right at the end of each loop
            if(np.isclose(Qleft.dot(BD).dot(Qright),A).all()):
                print(".", end=" ")
            else:
                print("!", end=" ")
            
    return (Qleft, BD, Qright)

