import numpy as np
import bidiagonal as bd

#test if the invariant stays the same for random matrix with differents size and for each iteration
def test_bidiagonal():
    print("test on the function bidiagonal with random vector of different sizes:")
    n = 10
    for i in range(0, n):
        A = np.random.rand(i, i)
        Qleft, BD, Qright = bd.bi_diagonal(A, 1)
        B = Qleft.dot(BD).dot(Qright)
        #test if all the value are equal within a tolerance(isclose function).
        if( np.isclose(Qleft.dot(BD).dot(Qright),A).all()):
            print("ok!")
        else:
            print("error in the return value")

test_bidiagonal()
