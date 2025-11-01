"""
Basic usage example for fkptjax.

This script demonstrates how to use the fkptjax package for
perturbation theory calculations.
"""

import jax.numpy as jnp
import fkptjax


def main() -> None:
    """Demonstrate basic fkptjax functionality."""
    print(f"fkptjax version: {fkptjax.__version__}")
    print()

    # Example 1: Scalar wavenumber
    k_scalar = 0.1
    result_scalar = fkptjax.compute_kernel(k_scalar, n=1)
    print(f"Kernel for k={k_scalar}: {result_scalar}")

    # Example 2: Array of wavenumbers
    k_array = jnp.logspace(-2, 0, 10)  # 0.01 to 1.0
    result_array = fkptjax.compute_kernel(k_array, n=1)
    print(f"\nKernel for k array:")
    for k, result in zip(k_array, result_array):
        print(f"  k={k:.4f}: {result:.4f}")

    # Example 3: Different kernel orders
    k = 0.5
    for n in [1, 2, 3]:
        result = fkptjax.compute_kernel(k, n=n)
        print(f"\nKernel order n={n}, k={k}: {result:.4f}")


if __name__ == "__main__":
    main()
