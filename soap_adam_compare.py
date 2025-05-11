# --- Modified soap_adam_compare.py to save final PINN parameters ---
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from jax import tree_util
from typing import NamedTuple, Callable, Tuple
import pickle

# ── 1) PINN + data ──────────────────────────────────────────
class PINN(nn.Module):
    hidden_sizes: Tuple[int,...] = (64,64,64)
    @nn.compact
    def __call__(self, x, t):
        h = jnp.stack([x,t], -1)
        for w in self.hidden_sizes:
            h = nn.tanh(nn.Dense(w)(h))
        return nn.Dense(1)(h).squeeze(-1)

def generate_data(key, N_f=10_000, N_i=1_000, N_b=1_000):
    k1,k2,k3 = jax.random.split(key, 3)
    x_f = jax.random.uniform(k1, (N_f,), minval=-1, maxval=1)
    t_f = jax.random.uniform(k2, (N_f,), minval= 0, maxval=1)
    x_i = jnp.linspace(-1,1,N_i);   t_i = jnp.zeros_like(x_i)
    u_i = -jnp.sin(jnp.pi * x_i)
    t_b = jax.random.uniform(k3, (N_b,), minval=0, maxval=1)
    x_lb, x_rb = -jnp.ones_like(t_b), jnp.ones_like(t_b)
    u_b = jnp.zeros_like(t_b)
    return (x_f,t_f), (x_i,t_i,u_i), (x_lb,x_rb,t_b,u_b)

def pde_residual(params, x, t, model, nu=0.01/jnp.pi):
    u_fn = lambda xx,tt: model.apply(params, xx, tt)
    u_x  = jax.grad(u_fn,    argnums=0)(x,t)
    u_t  = jax.grad(u_fn,    argnums=1)(x,t)
    u_xx = jax.grad(lambda xx,tt: jax.grad(u_fn, argnums=0)(xx,tt),
                    argnums=0)(x,t)
    return u_t + u_fn(x,t)*u_x - nu*u_xx

def loss_fn(params, batch, model):
    (x_f,t_f), (x_i,t_i,u_i), (x_lb,x_rb,t_b,u_b) = batch
    li = jnp.mean((model.apply(params,x_i,t_i)-u_i)**2)
    lb = jnp.mean((model.apply(params,x_lb,t_b)-u_b)**2 +
                  (model.apply(params,x_rb,t_b)-u_b)**2)
    res = jax.vmap(lambda xx,tt: pde_residual(params,xx,tt,model))(x_f,t_f)
    lf = jnp.mean(res**2)
    return li + lb + lf

# ── 2) SOAP trainer factory ─────────────────────────────────
class SoapState(NamedTuple):
    count: jnp.ndarray
    m:     any
    v:     any

def init_soap_state(params):
    zeros = lambda p: jnp.zeros_like(p)
    return SoapState(
      count=jnp.zeros([], jnp.int32),
      m=tree_util.tree_map(zeros, params),
      v=tree_util.tree_map(zeros, params),
    )

def make_soap_trainer(model,
                      lr: float = 1e-3,
                      b1: float = 0.9,
                      b2: float = 0.999,
                      eps: float = 1e-8):
    """Returns (init_fn, step_fn) where step_fn has signature
       (params, state, batch) -> (new_params, new_state, loss)."""
    @jax.jit
    def step(params, state, batch):
        (x_f,t_f), ic, bc = batch

        # total-loss + grad
        loss, grads_tot = jax.value_and_grad(lambda p: loss_fn(p, batch, model))(params)

        # per-sample residual grads
        g_res = jax.vmap(lambda xx,tt: jax.grad(
                     lambda pp,xx_,tt_: pde_residual(pp,xx_,tt_,model),
                     argnums=0)(params, xx, tt),
                 in_axes=(0,0))(x_f,t_f)

        # second-moment update
        v_new = tree_util.tree_map(lambda v_old, g: b2*v_old + (1-b2)*jnp.mean(g**2,axis=0),
                                   state.v, g_res)
        # momentum update
        m_new = tree_util.tree_map(lambda m_old, g: b1*m_old + (1-b1)*g,
                                   state.m, grads_tot)
        # precondition
        m_hat = tree_util.tree_map(lambda m,v: m/(jnp.sqrt(v)+eps), m_new, v_new)

        # apply step
        p_new = tree_util.tree_map(lambda w,mm: w - lr*mm, params, m_hat)
        s_new = SoapState(state.count+1, m_new, v_new)
        return p_new, s_new, loss

    def init(params):
        return init_soap_state(params)

    return init, step

# ── 3) Adam trainer factory (Optax) ────────────────────────
def make_adam_trainer(model, lr: float = 1e-3):
    optimizer = optax.adam(lr)

    @jax.jit
    def step(params, opt_state, batch):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, model))(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    def init(params):
        return optimizer.init(params)

    return init, step

# ── 4) Generic train loop ──────────────────────────────────
def train(rng: jax.random.PRNGKey,
          model: nn.Module,
          trainer_init: Callable,
          trainer_step: Callable,
          num_epochs: int = 2000,
          batch_size: int = 1024):
    # Init
    params   = model.init(rng, jnp.zeros((1,)), jnp.zeros((1,)))
    opt_state = trainer_init(params)

    # Data
    data = generate_data(rng)
    x_f, _  = data[0]
    num_batches = x_f.shape[0] // batch_size

    history = []
    for ep in range(1, num_epochs+1):
        perm = np.random.permutation(x_f.shape[0])
        epoch_loss = 0.0

        for i in range(num_batches):
            idx   = perm[i*batch_size:(i+1)*batch_size]
            batch = ((data[0][0][idx], data[0][1][idx]), data[1], data[2])
            params, opt_state, loss = trainer_step(params, opt_state, batch)
            epoch_loss += float(loss)

        epoch_loss /= num_batches
        history.append(epoch_loss)
        if ep==1 or ep%200==0:
            print(f"Epoch {ep:4d} | Loss {epoch_loss:.3e}")

    return history

# ── 5) Compare SOAP vs Adam ────────────────────────────────
def main():
    rng   = jax.random.PRNGKey(0)
    model = PINN()

    # SOAP
    soap_init, soap_step = make_soap_trainer(model)
    soap_params = model.init(rng, jnp.zeros((1,)), jnp.zeros((1,)))
    soap_state = soap_init(soap_params)
    data = generate_data(rng)
    for _ in range(2000):
        batch = data
        soap_params, soap_state, _ = soap_step(soap_params, soap_state, batch)

    # Save SOAP parameters
    with open("params_soap.pkl", "wb") as f:
        pickle.dump(soap_params, f)

    # Adam
    adam_init, adam_step = make_adam_trainer(model)
    adam_params = model.init(rng, jnp.zeros((1,)), jnp.zeros((1,)))
    adam_state = adam_init(adam_params)
    for _ in range(2000):
        batch = data
        adam_params, adam_state, _ = adam_step(adam_params, adam_state, batch)

    # Save Adam parameters
    with open("params_adam.pkl", "wb") as f:
        pickle.dump(adam_params, f)

if __name__ == "__main__":
    main()

