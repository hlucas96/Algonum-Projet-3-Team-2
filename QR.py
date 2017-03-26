import numpy as np

def transfo_QR(BD, Nmax):
    n =len(BD)
    U = np.identity(n)
    V = np.identity(n)
    S = BD

    for i in range (0 , Nmax):
        Q1, R1 = np.linalg.qr(np.transpose(S))
        Q2, R2 = np.linalg.qr(np.transpose(R1))
        S = R2
        U = U.dot(Q2)
        V = np.transpose(Q1).dot(V)
        print(S)

        USV = (U.dot(S)).dot(V)
        for j in range (0, n):
            for k in range (0, n):
                assert( abs(USV[j][k] - BD[j][k]) < 10**(-3))                    

    return U, S, V

BD = np.array([[1, 2, 0], [0, 3, 4], [0, 0, 5]])
U, S, V = transfo_QR(BD, 10**3)
