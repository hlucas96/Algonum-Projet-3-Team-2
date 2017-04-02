import numpy as np

#Algorithme USV
def transfo_USV(BD, Nmax):
    n = len(BD)
    U = np.identity(n)
    V = np.identity(n)
    S = BD

    for i in range (0 , Nmax):
        Q1, R1 = np.linalg.qr(np.transpose(S))
        Q2, R2 = np.linalg.qr(np.transpose(R1))
        S = R2
        U = U.dot(Q2)
        V = np.transpose(Q1).dot(V)
    USV = (U.dot(S)).dot(V)
    for j in range (0, n):
        for k in range (0, n):
            assert(abs(USV[j][k] - BD[j][k]) < 10**(-3))
    return U, S, V
    
#Decomposition QR pour une matrice bigonale
#Sous-decomposition QR avec des matrices 2x2
def transfo_QR_bigonale(BD):
    n = len(BD)
    Q = np.zeros((n,n))
    R = np.zeros((n,n))
    i = 0
    while (i + 1 <= n):
        A = BD[i:i+2, i:i+2]
        q, r= np.linalg.qr(A)
        Q[i:i+2,i:i+2] = q[0:2,0:2]
        R[i:i+2, i:i+2] = r[0:2, 0:2]
        i += 2
    return Q, R
 
#Algorithme USV améliorée
def transfo_USV_amelioree(BD, Nmax):
    n = len(BD)
    U = np.identity(n)
    V = np.identity(n)
    S = BD

    for i in range (0 , Nmax):
        Q1, R1 = transfo_QR_bigonale(np.transpose(S))
        Q2, R2 = transfo_QR_bigonale(np.transpose(R1))
        S = R2
        U = U.dot(Q2)
        V = np.transpose(Q1).dot(V)
    return U, S, V

    
#Trie decroissant des valeurs propres 
def ordre_vp(U, S, V):
    n = len(S)
    liste_vp = [i for i in np.diag(S)]
    liste_vp_decroissant = [i for i in np.diag(S)]
    liste_vp_decroissant = sorted(liste_vp_decroissant)[::-1]
    for i in range(1, n + 1):
        S[i - 1][i - 1] = liste_vp_decroissant[i - 1]
        for j in range(n):
            U[j][i-1] = U[j][i-1]*(liste_vp[i-1]/liste_vp_decroissant[i-1])
    return U, S

##############################TESTS###########################
    
#Test algorithme USV
def  test_USV(BD, Nmax):
    U, S, V = transfo_USV(BD, Nmax)
    USV = (U.dot(S)).dot(V)
    for j in range (0, n):
        for k in range (0, n):
            assert(abs(USV[j][k] - BD[j][k]) < 1)

    
#Test de comparaison entre l'algorithme de base et l'agorithme améliorée
def test_diff_USV():
    BD = np.array([[1, 2, 0, 0], [0, 3, 4,0], [0, 0, 5, 1], [0, 0, 0, 4]])
    U1, S1, V1 = transfo_USV(BD, 10**3)
    U2, S2, V2 = transfo_USV_amelioree(BD, 10**3)
    diff = (np.dot(U1,S1)).dot(V1) - (np.dot(U2,S2)).dot(V2)
    count = 0
    for i in range(len(BD)):
        for j in range(len(BD)):
            if (diff[i][j] > 10**(-5)):
                count += 1
    print("Tests entre les algorithmes USV :")
    print("Il y a ", count, "élément(s) de la surdiagonale perdu")
    

#Verification de la égalité USV = BD après tri
def test_ordre():
    BD = np.array([[1, 2, 0], [0, 3, 4], [0, 0, 5]])
    U, S, V = transfo_USV(BD, 10**3)
    U1, S1 = ordre_vp(U, S, V)
    USV = (np.dot(U1,S1)).dot(V)
    n = len(U)
    diff = USV - BD
    count = 0
    for i in range(len(BD)):
        for j in range(len(BD)):
            if (diff[i][j] > 10**(-5)):
                count += 1
    print("Test odonnner les valeurs propres :")
    if (count == 0):
        print("Success")
    else:
        print("Error")
    

test_ordre()
test_diff_USV()