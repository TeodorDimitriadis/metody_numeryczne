import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

def LDU(A):
    i,j  = np.indices(A.shape)

    L = A.copy()
    L[~ i < j] = 0

    D = A.copy()
    D[~ i == j] = 0

    U = A.copy()
    U[~ i > j] = 0

    return L, D, U

def gauss_seidl(A,b,max_iter,tol):
    L,D,U = LDU(A)

    n = len(b)
    x = np.zeros(n)
    D_inv = np.linalg.inv(D)

    for _ in range(max_iter):
        x_old = x.copy()
        for k in range(n):
            x[k] = D_inv[k,:] @ ( -L @ x -U @ x + b)

        if np.linalg.norm(x - x_old) < tol:
            return x

    return x


n = 20
A = np.zeros((n, n))
np.fill_diagonal(A, 4)
np.fill_diagonal(A[1:], -1)
np.fill_diagonal(A[:, 1:], -1)
A[0, n-1] = 1
A[n-1, 0] = 1

b = np.zeros(n)
b[-1] = 100

print(gauss_seidl(A,b,100,10 ** (-4)))
print('-------------------------------------')
print(np.linalg.solve(A,b))