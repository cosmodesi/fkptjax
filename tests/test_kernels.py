"""
Tests for perturbation theory kernels.
"""

import pytest
import jax.numpy as jnp
from fkptjax.kernels import compute_kernel


def test_compute_kernel_scalar() -> None:
    """Test kernel computation with scalar input."""
    k = 0.1
    result = compute_kernel(k, n=1)
    expected = 0.1
    assert jnp.isclose(result, expected)


def test_compute_kernel_array() -> None:
    """Test kernel computation with array input."""
    k = jnp.array([0.1, 0.2, 0.3])
    result = compute_kernel(k, n=1)
    expected = jnp.array([0.1, 0.2, 0.3])
    assert jnp.allclose(result, expected)


def test_compute_kernel_order() -> None:
    """Test kernel computation with different orders."""
    k = 2.0
    result_n1 = compute_kernel(k, n=1)
    result_n2 = compute_kernel(k, n=2)

    assert jnp.isclose(result_n1, 2.0)
    assert jnp.isclose(result_n2, 4.0)


def test_compute_kernel_invalid_order() -> None:
    """Test that invalid kernel order raises ValueError."""
    k = 0.1
    with pytest.raises(ValueError, match="Kernel order n must be >= 1"):
        compute_kernel(k, n=0)


def test_compute_kernel_negative_order() -> None:
    """Test that negative kernel order raises ValueError."""
    k = 0.1
    with pytest.raises(ValueError, match="Kernel order n must be >= 1"):
        compute_kernel(k, n=-1)
