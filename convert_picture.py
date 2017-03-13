import pylab as pl
import numpy as np

def picture_to_matrix(path):
    img_full = pl.imread(path)
    n = len(img_full);
    m = len(img_full[0])
    R,V,B = np.zeros((n,m)),np.zeros((n,m)),np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            R[i][j],V[i][j],B[i][j] = img_full[i][j][0],img_full[i][j][1],img_full[i][j][2]
    return (R,V,B)

def matrix_to_picture(R,V,B):
    n = len(R);
    m = len(R[0])
    M = np.zeros((n,m,3))
    for i in range(n):
        for j in range(m):
            M[i][j][0],M[i][j][1],M[i][j][2] = R[i][j],V[i][j],B[i][j]
    pl.imshow(M)

R,V,B = picture_to_matrix("essai.png")
matrix_to_picture(R, V, B)
