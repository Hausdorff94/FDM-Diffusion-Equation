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


def solver_theta(I, a, f, L, dt, F, T, theta=0.5, u_L=0, u_R=0,
                 user_action=None):
    """
    Full solver for the model problem using the theta-rule
    difference approximation in time (no restriction on F,
    i.e., the time step when theta >= 0.5).
    Vectorized implementation and sparse (tridiagonal)
    coefficient matrix.
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

    u = np.zeros(Nx+1)   # solution array at t[n+1]
    u_n = np.zeros(Nx+1)   # solution at t[n]

    # Representation of sparse matrix and right-hand side
    diagonal = np.zeros(Nx+1)
    lower = np.zeros(Nx)
    upper = np.zeros(Nx)
    b = np.zeros(Nx+1)

    # Precompute sparse matrix (scipy format)
    Fl = F*theta
    Fr = F*(1-theta)
    diagonal[:] = 1 + 2*Fl
    lower[:] = -Fl  # 1
    upper[:] = -Fl  # 1
    # Insert boundary conditions
    diagonal[0] = 1
    upper[0] = 0
    diagonal[Nx] = 1
    lower[-1] = 0

    diags = [0, -1, 1]
    A = scipy.sparse.diags(
        diagonals=[diagonal, lower, upper],
        offsets=[0, -1, 1], shape=(Nx+1, Nx+1),
        format='csr')
    # print A.todense()

    # Set initial condition
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Time loop
    for n in range(0, Nt):
        b[1:-1] = u_n[1:-1] + \
            Fr*(u_n[:-2] - 2*u_n[1:-1] + u_n[2:]) + \
            dt*theta*f(x[1:-1], t[n+1]) + \
            dt*(1-theta)*f(x[1:-1], t[n])
        b[0] = u_L
        b[-1] = u_R  # boundary conditions
        u[:] = scipy.sparse.linalg.spsolve(A, b)

        if user_action is not None:
            user_action(u, x, t, n+1)

        # Update u_n before next step
        u_n, u = u, u_n

        if n == 0:
            u_max = max(u_n) * 1.1

        figpath = 'images/sol-CN-at-' + str(n) + '.png'
        plt.plot(x, u_n)
        plt.ylim(0, u_max)
        plt.title('t = ' + str(n) + 's    ' + 'F = ' + str(F))
        plt.savefig(figpath)
        plt.clf()

        files.append(figpath)

    images = [imageio.imread(file) for file in files]
    imageio.mimwrite('sol-CN-F_' + str(F) + '.gif', images, fps=10)

    t1 = time.process_time()
    return u_n, x, t, t1-t0

def I(x):
    return np.exp(-0.5*((x-L/2.0)**2)/sigma**2)

def f(x, y):
    return 0


global a, L, F, sigma
a = 0.8
L = 5
dt = 0.01
F = 11
T = 1
sigma = 0.08

u, x, t, cpu = solver_theta(
    I=I, a=a, f=f, L=L, dt=dt, F=F, T=T)