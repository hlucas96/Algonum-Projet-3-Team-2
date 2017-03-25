import numpy as np
import pylab as pl

###Pour le moment ce fichier est inutilisable, les autres parties n'ayant pas été faites !

##Compresse un matrice carré M au rang k
def compress(M, k):
    S = M;
    ## U, S, V = decompo_usv(M)
    n = len(M)
    if (k < n):
        for i in range(k + 1, n + 1):
            S[i][i] = 0
    res = S
    ##res = np.dot(np.dot(U,S),V)
    return res

##Définition de la norme comme l'élément maximal de la matrice
def norme(M):
    n = len(M)
    p = len(M[0])
    sup = 0
    for i in range(n):
        for j in range(p):
            if abs(M[i][j])>sup:
                sup = abs(M[i][j])
    return sup

##Trace l'efficacité en fonction du rang pour une matrice donnée
def efficacite(M):
    n = len(M)
    x = range(1, n + 1)
    distances = []
    for i in x:
        M_comp = compress(M, k)
        distances += norme(M - M_comp)
    pl.plot(x,distances)
    pl.xlabel("Rang de compression k")
    pl.ylabel("Efficacite de la compression")
    pl.show()
