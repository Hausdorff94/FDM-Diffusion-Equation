#!/usr/bin/env python
"""
Functions for solving a 1D diffusion equations of simplest types
(constant coefficient, no source term):

      u_t = a*u_xx on (0,L)

with boundary conditions u=0 on x=0,L, for t in (0,T].
Initial condition: u(x,0)=I(x).

The following naming convention of variables are used.

===== ==========================================================
Name  Description
===== ==========================================================
Nx    The total number of mesh cells; mesh points are numbered
      from 0 to Nx.
F     The dimensionless number a*dt/dx**2, which implicitly
      specifies the time step.
T     The stop time for the simulation.
I     Initial condition (Python function of x).
a     Variable coefficient (constant).
L     Length of the domain ([0,L]).
x     Mesh points in space.
t     Mesh points in time.
n     Index counter in time.
u     Unknown at current/new time level.
u_n   u at the previous time level.
dx    Constant mesh spacing in x.
dt    Constant mesh spacing in t.
===== ==========================================================
"""
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import imageio

#import scitools.std as plt


def solver_FE_simple(I, a, f, L, dt, F, T):
    files = []
    t0 = time.process_time()  # For measuring the CPU time

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

    # Set initial condition u(x,0) = I(x)
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    for n in range(0, Nt):
        # Compute u at inner mesh points
        for i in range(1, Nx):
            u[i] = u_n[i] + F*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) + \
                dt*f(x[i], t[n])

        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0

        # Switch variables before next step
        u_n, u = u, u_n

        if n == 0:
            u_max = max(u_n) * 1.1

        aux = max(u_n)
        if aux > u_max:
            u_max = aux*1.1

        figpath = 'images/sol-exp-at-' + str(n) + '.png'
        plt.plot(x, u_n)
        plt.ylim(0, u_max)
        plt.title('t = ' + str(n) + 's    ' + 'F = ' + str(F))
        plt.savefig(figpath)
        plt.clf()

        files.append(figpath)

    t1 = time.process_time()

    images = [imageio.imread(file) for file in files]
    imageio.mimwrite('sol-exp-F_' + str(F) + '.gif', images, fps=10)

    return u_n, x, t, t1-t0  # u_n holds latest u


def solver_FE(I, a, f, L, dt, F, T,
              user_action=None, version='scalar'):
    """
    Vectorized implementation of solver_FE_simple.
    """
    t0 = time.clock()  # for measuring the CPU time

    Nt = int(round(T/float(dt)))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    # Make sure dx and dt are compatible with x and t
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u = np.zeros(Nx+1)   # solution array
    u_n = np.zeros(Nx+1)   # solution at t-dt

    # Set initial condition
    for i in range(0, Nx+1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    for n in range(0, Nt):
        # Update all inner points
        if version == 'scalar':
            for i in range(1, Nx):
                u[i] = u_n[i] +\
                    F*(u_n[i-1] - 2*u_n[i] + u_n[i+1]) +\
                    dt*f(x[i], t[n])

        elif version == 'vectorized':
            u[1:Nx] = u_n[1:Nx] +  \
                F*(u_n[0:Nx-1] - 2*u_n[1:Nx] + u_n[2:Nx+1]) +\
                dt*f(x[1:Nx], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0
        if user_action is not None:
            user_action(u, x, t, n+1)

        # Switch variables before next step
        u_n, u = u, u_n

    t1 = time.clock()
    return t1-t0


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

u, x, t, cpu = solver_FE_simple(
    I=I, a=a, f=f, L=L, dt=dt, F=F, T=T)
