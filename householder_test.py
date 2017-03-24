import time
import matplotlib.pyplot as plt
import numpy as np

import householder as hh

def plot_householder():
    max_size = 1000
    xaxis = np.arange(1, max_size, 5)
    execute_time_n = []
    execute_time_o = []
    I = np.zeros(max_size)
    I.shape = (max_size, 1)
    J = np.ones(max_size)
    J.shape = (max_size, 1)
    X = np.ones((max_size,max_size))
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
    I = np.array([[1], [1], [1]])
    J = np.array([[0], [0], [1]])
    X = np.array([[3, 4, 7], [4, 7, 9], [0, 0, 0]])

    print("householder test:")
    print("householder matrix:")
    print(hh.householder(I, J))
    print("apply to:")
    print(X)

    print("non-optimized householder:")
    print(hh.mul_householder(X, I, J))
    print("optimized householder:")
    print(hh.mul_householder_optimized(X, I, J))

    print("plot of the execute time ongoing...")
    plot_householder()

householder_test()
