#!/usr/bin/env python

import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import imageio


def solver_BE_simple(I, a, f, L, dt, F, T):
    """
    Simplest expression of the computational algorithm
    for the Backward Euler method, using explicit Python loops
    and a dense matrix format for the coefficient matrix.
    """
    files = []

    t0 = time.process_time()  # for measuring the CPU time

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)

    # Data structures for the linear system
    A = np.zeros((Nx+1, Nx+1))
    b = np.zeros(Nx+1)

    for i in range(1, Nx):
        A[i, i-1] = -F
        A[i, i+1] = -F
        A[i, i] = 1 + 2*F
    A[0, 0] = A[Nx, Nx] = 1

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    for n in range(0, Nt):
        # Compute b and solve linear system
        for i in range(1, Nx):
            b[i] = u_n[i] + dt*f(x[i], t[n+1])
        b[0] = b[Nx] = 0
        u[:] = np.linalg.solve(A, b)

        # Update u_n before next step
        u_n, u = u, u_n

        if n == 0:
            u_max = max(u_n) * 1.1

        figpath = 'images/sol-imp-at-' + str(n) + '.png'
        plt.plot(x, u_n)
        plt.ylim(0, u_max)
        plt.title('t = ' + str(n) + 's    ' + 'F = ' + str(F))
        plt.savefig(figpath)
        plt.clf()

        files.append(figpath)

    t1 = time.process_time()

    images = [imageio.imread(file) for file in files]
    imageio.mimwrite('sol-imp-F_' + str(F) +'.gif', images, fps=10)

    return u_n, x, t, t1-t0


def I(x):
    return np.exp(-0.5*((x-L/2.0)**2)/sigma**2)


def f(x, y):
    return 0


global a, L, F, sigma
a = 0.8
L = 5
dt = 0.01
F = 0.55
T = 1
sigma = 0.08

u, x, t, cpu = solver_BE_simple(
    I=I, a=a, f=f, L=L, dt=dt, F=F, T=T)
