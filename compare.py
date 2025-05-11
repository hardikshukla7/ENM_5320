import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import linen as nn
import pickle

# Load FDM solution data (x, t, U)
data = np.load("fdm_solution.npz")
U_fdm = data["U"]
x = data["x"]
t = data["t"]
nt, nx = U_fdm.shape

# Create meshgrid for PINN evaluation
X, T = np.meshgrid(x, t)
X_flat = jnp.ravel(X)
T_flat = jnp.ravel(T)

# Define PINN model
class PINN(nn.Module):
    hidden_sizes: tuple = (64, 64, 64)

    @nn.compact
    def __call__(self, x, t):
        xt = jnp.stack([x, t], axis=-1)
        h = xt
        for w in self.hidden_sizes:
            h = nn.tanh(nn.Dense(w)(h))
        return nn.Dense(1)(h).squeeze(-1)

# Instantiate model
model = PINN()

# Load trained parameters
with open("params_adam.pkl", "rb") as f:
    params_adam = pickle.load(f)

with open("params_soap.pkl", "rb") as f:
    params_soap = pickle.load(f)

# Evaluate model predictions
u_pred_adam = model.apply(params_adam, X_flat, T_flat)
u_pred_soap = model.apply(params_soap, X_flat, T_flat)

# Reshape into 2D time-space grid
U_adam = np.array(u_pred_adam).reshape(nt, nx)
U_soap = np.array(u_pred_soap).reshape(nt, nx)

# Compute error maps
err_adam = U_fdm - U_adam
err_soap = U_fdm - U_soap

# Plot both error maps
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

im1 = axes[0].imshow(err_adam, extent=[x.min(), x.max(), t.min(), t.max()],
                     aspect='auto', origin='lower', cmap='coolwarm')
axes[0].set_title("Error: FDM - Adam PINN")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(err_soap, extent=[x.min(), x.max(), t.min(), t.max()],
                     aspect='auto', origin='lower', cmap='coolwarm')
axes[1].set_title("Error: FDM - SOAP PINN")
axes[1].set_xlabel("x")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
