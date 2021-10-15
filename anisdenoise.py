import numpy as np
import copy

def anisdenoise(im):

    ims = im
    m, n = ims.shape
    gx, gy = np.gradient(im)
    grad_square = np.multiply(gx,gx)+ np.multiply(gy, gy)
    g = np.zeros((m,n))
    for i in range(0, m):
        for j in range(0, n):
            g[i, j] = 1 / (1 + 1000 * grad_square[i, j])

    A = np.zeros((m,n))
    mu = 1e-3
    tau = 5e-1
    iter = 100
    hx = 1/(m-1)
    hy = 1/(n-1)
    cx = mu / (hx**2)
    cy = mu / (hy **2)
    A = copy.deepcopy(g)
    A[0:m-2,:] = (1/2) * (g[1:m-1,:] + A[0:m-2,:])
    B = copy.deepcopy(g)
    B[1:m-1,:] = (1/2) * (g[0:m-2,:] + B[1:m-1,:])
    C = copy.deepcopy(g)
    C[:,0:n-2] = (1/2) * (g[:,1:n-1] + C[:,0:n-2])
    D = copy.deepcopy(g)
    D[:,1:n-1] = (1/2) * (g[:,0:n-2] + D[:,1:n-1])

    A = cx * A
    B = cx * B
    C = cy * C
    D = cy * D

    u = im
    num = np.zeros((m,n))
    denom = A + B+ C +D
    denom = denom +tau
    newu = copy.deepcopy(u)

    for k in range(0,iter):

        for i in range(0,m):
            for j in range(0,n):
                if i ==0:
                    if j ==0:
                        num[i,j] = (A[i,j] + B[i,j] )*u[i+1,j]+ (C[i,j]+D[i,j])*u[i,j+1]
                    elif j ==n-1:
                        num[i,j] = (A[i,j]+ B[i,j])* u[i+1,j]+ (C[i,j]+D[i,j])*u[i,j-1]
                    else:
                        num[i,j] = (A[i,j]+ B[i,j])* u[i+1,j]+ C[i,j] * u[i,j+1] + D[i,j] * u[i,j-1]

                elif i ==m-1:
                    if j ==0:
                        num[i,j] = (A[i, j] + B[i, j]) * u[i -1, j] + (C[i, j] + D[i, j]) * u[i, j + 1]
                    elif j ==n-1:
                        num[i,j] = (A[i, j] + B[i, j]) * u[i - 1, j] + (C[i, j] + D[i, j]) * u[i, j - 1]
                    else:
                        num[i,j] = (A[i, j] + B[i, j]) * u[i - 1, j] + C[i, j] * u[i, j + 1] + D[i, j] * u[i, j - 1]
                else:
                    if j ==0:
                        num[i, j] = A[i, j] * u[i+1, j] + B[i, j] * u[i - 1, j] + (C[i, j]+ D[i, j]) * u[i, j + 1]
                    elif j == n-1:
                        num[i,j] = A[i, j]*u[i+1,j] + B[i, j] * u[i - 1, j] + C[i, j]*u[i,j-1] + D[i, j] * u[i, j - 1]
                    else:
                        num[i,j] = A[i, j]*u[i+1,j] + B[i, j] * u[i - 1, j] + C[i, j] * u[i, j + 1] + D[i, j] * u[i, j - 1]

                newu[i,j] = num[i,j] / denom[i,j]

    return newu