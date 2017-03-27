import numpy as np
import pylab as pl
import QR as qr
import convert_picture as conv


##Compresse un matrice carre M au rang k
def compress(M, k):
    n = len(M)
    U, S, V = qr.transfo_QR(M, n)
    if (k < n):
        for i in range(k + 1, n + 1):
            S[i][i] = 0
    res = np.dot(np.dot(U, S), V)
    return res

##Definition de la norme comme l'element maximal de la matrice
def norme(M):
    n = len(M)
    p = len(M[0])
    sup = 0
    for i in range(n):
        for j in range(p):
            if abs(M[i][j])>sup:
                sup = abs(M[i][j])
    return sup

##Trace l'efficacite en fonction du rang pour une matrice donnee
def efficacite(M):
    n = len(M)
    x = range(1, n + 1, 100)
    distances = []
    for k in x:
        M_comp = compress(M, k)
        print("done")
        distances += norme(M - M_comp)
    pl.plot(x,distances)
    pl.xlabel("Rang de compression k")
    pl.ylabel("Efficacite de la compression")
    pl.show()


R, V, B = conv.picture_to_matrix("essai2.png")
print(qr.transfo_QR(R, 500))
efficacite(R)
for k in [1, 225, 375, 500]:
    conv.matrix_to_picture(compress(R, k), compress(V, k), compress(B, k))
