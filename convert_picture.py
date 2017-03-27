import matplotlib.pyplot as pl
import numpy as np
import matplotlib.image as img

def picture_to_matrix(path):
    ##Return 3 matrixs corresponding to the color Red Green Blue of the picture in path
    img_full = img.imread(path)
    n = len(img_full);
    m = len(img_full[0])
    R,V,B = np.zeros((n,m)),np.zeros((n,m)),np.zeros((n,m))

    for i in range(n):
        for j in range(m):
            R[i][j],V[i][j],B[i][j] = img_full[i][j][0],img_full[i][j][1],img_full[i][j][2]
    return (R,V,B)

def matrix_to_picture(R,V,B):
    ##Print the picture corresponding to the colours of the matrix R, V, B
    n = len(R);
    m = len(R[0])
    M = np.zeros((n,m,3))
    for i in range(n):
        for j in range(m):
            M[i][j][0],M[i][j][1],M[i][j][2] = R[i][j],V[i][j],B[i][j]
    pl.imshow(M)
    pl.show()
