import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx = 201                      # Spatial resolution
x = np.linspace(-1, 1, nx)    # Spatial domain
dx = x[1] - x[0]
nu = 0.01 / np.pi             # Viscosity
dt = 0.0005                   # Time step
T = 1.0                       # Final time
nt = int(T / dt)              # Number of time steps
t = np.linspace(0, T, nt)     # Time domain

# Initial condition
u = -np.sin(np.pi * x)
u[0] = 0
u[-1] = 0

# Storage for u(x,t)
U = np.zeros((nt, nx))
U[0, :] = u

# Time stepping loop
for n in range(1, nt):
    un = u.copy()
    for i in range(1, nx - 1):
        u[i] = (un[i]
                - dt / (2 * dx) * un[i] * (un[i + 1] - un[i - 1])
                + nu * dt / dx**2 * (un[i + 1] - 2 * un[i] + un[i - 1]))
    
    # Enforce boundary conditions
    u[0] = 0
    u[-1] = 0

    U[n, :] = u

# Plotting the heatmap
plt.figure(figsize=(10, 6))
plt.imshow(U, extent=[-1, 1, 0, T], aspect='auto', origin='lower',
           cmap='viridis')
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title("Heatmap of 1D Burgers' Equation Solution u(x,t)")
plt.show()

import pickle

# Save solution for later comparison
with open("fdm_solution.pkl", "wb") as f:
    pickle.dump({'U': U, 'x': x, 't': t}, f)
