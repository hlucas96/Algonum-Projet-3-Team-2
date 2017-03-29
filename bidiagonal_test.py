import numpy as np
import bidiagonal as bd

#test if the invariant stays the same for random matrix with differents size and for each iteration
def test_bidiagonal():
    print("test on the function bidiagonal with random vector of different sizes:")
    print("invariant stay valid")
    print("BD matrix is bidiagonal")
    n = 10
    for i in range(0, n):
        A = np.random.rand(i, i)
        Qleft, BD, Qright = bd.bi_diagonal(A, 1)
        B = Qleft.dot(BD).dot(Qright)
        #test if all the value except the bidiagonal are equal to zero.
        for k in range (0, i):
            for l in range (0, i):
                #non-diagonal term
                if(k != l):
                    #not an higher extradiagonal term.
                    if(k + 1 != l):
                        if(np.isclose(BD[k][l], 0)):
                            print(".", end= " ")
                        else:
                            print("!", end=" ")
                        
        #test if all the value are equal within a tolerance(isclose function).
        if( np.isclose(Qleft.dot(BD).dot(Qright),A).all()):
            print("ok")
        else:
            print("error in the return value")

test_bidiagonal()
