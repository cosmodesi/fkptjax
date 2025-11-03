"""
fkptjax: Perturbation theory calculations for LCDM and Modified Gravity theories.

This package implements perturbation theory calculations using fk-Kernels
with JAX for high-performance automatic differentiation and JIT compilation.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("fkptjax")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
