"""Core mathematical transforms: adstock and Hill saturation."""

from jax import lax


def adstock_geometric(x, alpha, init=0.0):
    """Geometric decay adstock: s_t = x_t + alpha * s_{t-1}."""

    def step(carry, x_t):
        carry = x_t + alpha * carry
        return carry, carry

    _, s = lax.scan(step, init, x)
    return s


def hill(x, A, k, n):
    """Hill saturation: y = A * x^n / (k^n + x^n)."""
    return A * (x**n) / (k**n + x**n + 1e-12)


def hill_matrix(s, A, k, n):
    """Vectorized Hill over K components: (T,) x (K,) -> (T, K)."""
    s_col = s[:, None]
    return (
        A[None, :]
        * (s_col ** n[None, :])
        / (k[None, :] ** n[None, :] + s_col ** n[None, :] + 1e-12)
    )
