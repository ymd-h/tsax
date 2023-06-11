import jax
import jax.numpy as jnp
import pandas as pd

def generate():
    L = 30000
    d = 3

    x = jnp.arange(L)

    raw = (jnp.zeros((L, d))
         .at[:,0].add(jnp.sin(jnp.pi * 0.75 * x))
         .at[:,0].add(0.3 * jnp.cos(jnp.pi * 0.4 * x))
         .at[:,1].add(jnp.sin(-jnp.pi * 0.01 * x))
         .at[:,1].add(jnp.cos(jnp.pi * 1.77 * x))
         .at[:,2].add(jnp.sin(jnp.pi / (x + 1)))
         .at[:,2].add(0.4 * jnp.sin(jnp.pi * 0.1 * x * x)))

    data = pd.DataFrame(raw,
                        index=pd.date_range(start="2023/06/01", freq="1D", periods=L),
                        columns=[f"col-{i}" for i in range(d)])

    data.to_csv("sample-with-index.csv", index=True)
    data.to_csv("sample-without-index.csv", index=False)

if __name__ == "__main__":
    generate()
