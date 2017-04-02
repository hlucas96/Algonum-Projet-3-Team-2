import numpy as np
import pylab as pl
import QR as qr
import convert_picture as conv
import bidiagonal as bi


##Compresse un matrice carre M au rang k
def compress(M, k):
    n = len(M)
    QL, BD, QR = bi.bi_diagonal(M, 0)
    U, S, V = qr.transfo_USV(BD, 10**2)
    U, S = qr.ordre_vp(U, S)
    if (k < n):
        for i in range(k + 1, n):
            S[i][i] = 0
    res = np.dot(QL,np.dot(U, np.dot(S, np.dot(V, QR))))
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
    QL, BD, QR = bi.bi_diagonal(M, 0)
    U, S, V = qr.transfo_USV(BD, 10**2)
    U, S = qr.ordre_vp(U, S)
    x = range(n-1, 0, -1)
    distances = []
    for i in x:
        S[i][i] = 0
        distances += [norme(M - M_comp)]
    pl.plot(mirror(x), mirror(distances))
    pl.xlabel("Rang de compression k")
    pl.ylabel("Efficacite de la compression")
    pl.show()


R, V, B = conv.picture_to_matrix("essai.png")
k = 25
conv.matrix_to_picture(compress(R, k), compress(V, k), compress(B, k))
efficacite(R)
#for k in [1, 50, 75, 100]:
#    conv.matrix_to_picture(compress(R, k), compress(V, k), compress(B, k))
