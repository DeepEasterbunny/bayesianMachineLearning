import jax.numpy as jnp

sigmoid = lambda x: 1./(1 + jnp.exp(-x))
log_npdf = lambda x, m, v: -(x-m)**2/(2*v) - 0.5*jnp.log(2*jnp.pi*v)
softmax = lambda x: jnp.exp(x) / (jnp.sum(jnp.exp(x)))