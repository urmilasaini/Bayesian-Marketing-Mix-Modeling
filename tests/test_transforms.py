"""Tests for core transforms.

Verifies adstock and hill functions work correctly.
"""

import jax.numpy as jnp
import numpy as np

from hill_mixture_mmm.transforms import adstock_geometric, hill, hill_matrix


class TestAdstockGeometric:
    """Tests for adstock_geometric transform."""

    def test_alpha_zero_returns_input(self):
        """With alpha=0, no carryover - output equals input."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = adstock_geometric(x, alpha=0.0)
        np.testing.assert_allclose(result, x, rtol=1e-5)

    def test_alpha_one_returns_cumsum(self):
        """With alpha=1, full persistence - output is cumulative sum."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = adstock_geometric(x, alpha=1.0)
        expected = jnp.cumsum(x)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_alpha_half_geometric_decay(self):
        """With alpha=0.5, verify geometric decay pattern."""
        x = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = adstock_geometric(x, alpha=0.5)
        expected = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_output_shape_matches_input(self):
        """Output should have same shape as input."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = adstock_geometric(x, alpha=0.5)
        assert result.shape == x.shape


class TestHill:
    """Tests for hill saturation function."""

    def test_zero_input_returns_zero(self):
        """hill(0) should be 0 for any parameters."""
        result = hill(0.0, A=10.0, k=5.0, n=2.0)
        assert abs(result) < 1e-10

    def test_at_half_saturation(self):
        """At x=k, effect should be A/2."""
        A, k, n = 10.0, 5.0, 2.0
        result = hill(k, A, k, n)
        expected = A / 2
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_input_approaches_asymptote(self):
        """As x -> inf, effect -> A."""
        A, k, n = 10.0, 5.0, 2.0
        result = hill(1e10, A, k, n)
        np.testing.assert_allclose(result, A, rtol=1e-3)

    def test_monotonically_increasing(self):
        """Hill function should be monotonically increasing."""
        x = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
        result = hill(x, A=10.0, k=5.0, n=2.0)
        diffs = jnp.diff(result)
        assert jnp.all(diffs >= 0), "Hill function should be monotonically increasing"

    def test_steepness_increases_with_n(self):
        """Higher n should give steeper curve."""
        x = 5.0
        low_n = hill(x * 0.5, A=10.0, k=x, n=1.0)
        high_n = hill(x * 0.5, A=10.0, k=x, n=4.0)
        assert low_n > high_n


class TestHillMatrix:
    """Tests for vectorized hill_matrix function."""

    def test_output_shape(self):
        """Output should be (T, K) for T inputs and K components."""
        s = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        A = jnp.array([10.0, 20.0, 30.0])
        k = jnp.array([2.0, 3.0, 4.0])
        n = jnp.array([1.5, 2.0, 2.5])

        result = hill_matrix(s, A, k, n)
        assert result.shape == (5, 3)

    def test_matches_scalar_hill(self):
        """Matrix version should match scalar hill for each component."""
        s = jnp.array([1.0, 2.0, 3.0])
        A = jnp.array([10.0, 20.0])
        k = jnp.array([2.0, 3.0])
        n = jnp.array([1.5, 2.0])

        result = hill_matrix(s, A, k, n)

        for t in range(len(s)):
            for j in range(len(A)):
                expected = hill(s[t], A[j], k[j], n[j])
                np.testing.assert_allclose(
                    result[t, j], expected, rtol=1e-5, err_msg=f"Mismatch at t={t}, j={j}"
                )
