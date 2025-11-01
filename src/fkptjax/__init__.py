"""
fkptjax: Perturbation theory calculations for LCDM and Modified Gravity theories.

This package implements perturbation theory calculations using fk-Kernels
with JAX for high-performance automatic differentiation and JIT compilation.
"""

from fkptjax.kernels import compute_kernel

__version__ = "0.1.0"
__all__ = ["__version__", "compute_kernel"]
