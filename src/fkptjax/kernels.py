"""
Perturbation theory kernels for cosmological calculations.
"""

from typing import Union
import jax.numpy as jnp
from jax import Array


def compute_kernel(
    k: Union[float, Array],
    n: int = 1
) -> Union[float, Array]:
    """
    Compute a simple perturbation theory kernel.

    This is a placeholder function demonstrating the package structure.
    Replace with actual fk-kernel calculations.

    Parameters
    ----------
    k : float or Array
        Wavenumber(s) in units of h/Mpc.
    n : int, optional
        Kernel order. Default is 1.

    Returns
    -------
    float or Array
        Computed kernel value(s).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from fkptjax.kernels import compute_kernel
    >>> k = jnp.array([0.1, 0.2, 0.3])
    >>> result = compute_kernel(k, n=1)
    """
    if n < 1:
        raise ValueError("Kernel order n must be >= 1")

    # Placeholder implementation
    return jnp.power(k, n)
