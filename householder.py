import numpy as np
import matplotlib.pyplot as plt
import time

def householder(X, Y):
    U = (X - Y) / np.linalg.norm(X - Y)
    I = np.identity(len(U))
    H = I - 2.0 * U.dot(U.T)
    return H

def mul_householder(X, I, J):
    H = householder(I, J)
    return H.dot(X)
        
def mul_householder_optimized(X, I, J):
    U = (I - J) / np.linalg.norm(I - J)
    S = X - 2 * U.dot(U.T.dot(X))
    return S

def plot_householder():
    max_size = 500
    xaxis = np.arange(1, max_size, 10)
    execute_time_n = []
    execute_time_o = []
    I = np.zeros(max_size)
    J = np.ones(max_size)
    X = np.ones(max_size)
    for h in xaxis:
        t_a = time.time()
        mul_householder(X[:h],I[:h], J[:h])
        t_b = time.time()
        mul_householder_optimized(X[:h],I[:h], J[:h])
        t_c = time.time()
        execute_time_n.append(t_b - t_a)
        execute_time_o.append(t_c - t_b)

        
    l_h_n, = plt.plot(xaxis, execute_time_n, linewidth = 1.0)
    l_h_o, = plt.plot(xaxis, execute_time_o, linewidth = 1.0)
    plt.legend([l_h_n, l_h_o], ["householder", "householder_optimized"])
    plt.show()
    
I = np.array([[3], [4], [0]])
J = np.array([[0], [0], [5]])
X = np.array([[3, 4, 7], [4, 7, 9], [0, 0, 0]])
print(mul_householder(X, I, J))
print(mul_householder_optimized(X, I, J))
plot_householder()
