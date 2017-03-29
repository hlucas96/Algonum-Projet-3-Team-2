import time
import matplotlib.pyplot as plt
import numpy as np

import householder as hh


#plot execute time of optmized and non-optimized function
def plot_householder():
    max_size = 1000
    xaxis = np.arange(1, max_size, 5)
    execute_time_n = []
    execute_time_o = []
    I = np.random.randint(max_size, size =(max_size,1))
    J = np.zeros((max_size, 1))
    J[0] = hh.norme(I)
    X = np.random.randint(max_size, size = (max_size, max_size))
    for h in xaxis:
        t_a = time.time()
        hh.mul_householder(X[0:h,0:h],I[:h], J[:h])
        t_b = time.time()
        hh.mul_householder_optimized(X[0:h, 0:h],I[:h], J[:h])
        t_c = time.time()
        execute_time_n.append(t_b - t_a)
        execute_time_o.append(t_c - t_b)

        
    l_h_n, = plt.plot(xaxis, execute_time_n, linewidth = 1.0)
    l_h_o, = plt.plot(xaxis, execute_time_o, linewidth = 1.0)
    plt.legend([l_h_n, l_h_o], ["householder", "householder_optimized"])
    plt.show()



def householder_test():
    print("test on random vector with different sizes of:")
    print("householder function")
    print("mul_householder function")
    print("mul_householder_optimized function")

    n = 10
    for i in range(2, n):
        #create random vector
        I = np.random.rand(i)
        I.shape = (i, 1)
        
        J = np.zeros(i)
        J[i-1] = hh.norme(I)
        J.shape = (i, 1)

        #test function householder
        if (np.isclose((hh.householder(I, J)).dot(I), J).all()):
            print(".", end= " ")
        else:
            print("!", end= " ")

        #test function mul_householder
        if (np.isclose(hh.mul_householder(I, I, J), J).all()):
            print(".", end= " ")
        else:
            print("!", end= " ")

        #test function mul_householder_optimized
        if (np.isclose(hh.mul_householder_optimized(I, I, J), J).all()):
            print(".", end= " ")
        else:
            print("!", end= " ")

    print()
    print("plot of the execute time ongoing...")
    plot_householder()

householder_test()
