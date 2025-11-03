from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from fkptjax.types import Float64NDArray


@dataclass
class InputParams:
    """Input parameters for k-functions calculations"""
    # Cosmological parameters
    f0: float  # Growth rate at k→0

    # Output k-grid parameters (specify desired k grid for computed KFunctions)
    kmin: float  # Minimum k value for output k-functions grid [h/Mpc]
    kmax: float  # Maximum k value for output k-functions grid [h/Mpc]
    Nk: int      # Number of k points in output k-functions grid

    # Numerical integration parameters
    nquadSteps: int  # Number of quadrature steps for k-integration
    NQ: int          # Number of Gauss-Legendre points for Q-functions
    NR: int          # Number of Gauss-Legendre points for R-functions

    # SPT kernel constants
    KA_LCDM: float   # Kernel constant A
    KAp_LCDM: float  # Kernel constant Ap
    KR1_LCDM: float  # Kernel constant CFD3
    KR1p_LCDM: float # Kernel constant CFD3'

    # Variance and damping integrals
    sigma2v: float  # Velocity dispersion σ²_v

@dataclass
class InputArrays:
    """Input arrays for k-functions calculations"""
    k_in: Float64NDArray   # k values [h/Mpc]
    Pk_in: Float64NDArray  # Linear power spectrum (wiggle) [(Mpc/h)³]
    Pk_nw_in: Float64NDArray  # Linear power spectrum (no-wiggle) [(Mpc/h)³]
    f_in: Float64NDArray   # Growth rate f(k)

@dataclass
class KFunctions:
    """Expected k-functions output (27 arrays + k-grid)"""
    k: Float64NDArray  # Output k-grid (computed from kmin/kmax/Nk)
    # P22 components
    P22dd: Float64NDArray
    P22du: Float64NDArray
    P22uu: Float64NDArray
    # P13 components
    P13dd: Float64NDArray
    P13du: Float64NDArray
    P13uu: Float64NDArray
    # RSD A-terms
    I1udd1A: Float64NDArray
    I2uud1A: Float64NDArray
    I2uud2A: Float64NDArray
    I3uuu2A: Float64NDArray
    I3uuu3A: Float64NDArray
    # RSD D-terms (B+C-G)
    I2uudd1BpC: Float64NDArray
    I2uudd2BpC: Float64NDArray
    I3uuud2BpC: Float64NDArray
    I3uuud3BpC: Float64NDArray
    I4uuuu2BpC: Float64NDArray
    I4uuuu3BpC: Float64NDArray
    I4uuuu4BpC: Float64NDArray
    # Bias terms
    Pb1b2: Float64NDArray
    Pb1bs2: Float64NDArray
    Pb22: Float64NDArray
    Pb2s2: Float64NDArray
    Ps22: Float64NDArray
    Pb2theta: Float64NDArray
    Pbs2theta: Float64NDArray
    # Additional
    sigma32PSL: Float64NDArray
    pkl: Float64NDArray  # Linear P(k) on output grid

@dataclass
class KFunctionsSnapshot:
    """Test data snapshot loaded from .npz file"""
    # Parameters
    params: InputParams
    # Input arrays
    arrays: InputArrays
    # Expected outputs
    kfuncs_wiggle: KFunctions
    kfuncs_nowiggle: KFunctions

    def print_params(self) -> None:
        """Pretty-print input parameters with descriptive labels and units."""
        p = self.params

        print("Input Parameters")
        print("=" * 60)

        print("\nCosmology:")
        print(f"  Growth rate at k→0 (f0)                : {p.f0:.6f}")

        print("\nOutput k-grid (for computed KFunctions):")
        print(f"  Minimum k (kmin)                       : {p.kmin:.6f} h/Mpc")
        print(f"  Maximum k (kmax)                       : {p.kmax:.6f} h/Mpc")
        print(f"  Number of k points (Nk)                : {p.Nk}")

        print("\nNumerical Integration:")
        print(f"  Quadrature steps for k-integration     : {p.nquadSteps}")
        print(f"  Gauss-Legendre points for Q-functions  : {p.NQ}")
        print(f"  Gauss-Legendre points for R-functions  : {p.NR}")

        print("\nSPT Kernel Constants (LCDM):")
        print(f"  Kernel constant A (KA_LCDM)            : {p.KA_LCDM:.6f}")
        print(f"  Kernel constant Ap (KAp_LCDM)          : {p.KAp_LCDM:.6f}")
        print(f"  Kernel constant CFD3 (KR1_LCDM)        : {p.KR1_LCDM:.6f}")
        print(f"  Kernel constant CFD3' (KR1p_LCDM)      : {p.KR1p_LCDM:.6f}")

        print("\nVariance and Damping:")
        print(f"  Velocity dispersion σ²_v (sigma2v)     : {p.sigma2v:.6f} (Mpc/h)²")

        print("=" * 60)


def get_default_snapshot_path() -> str:
    """Get path to default test data snapshot file.

    Returns:
        Path to tests/data/test_data.npz relative to package root
    """
    # Get the directory containing this file (src/fkptjax/)
    package_dir = Path(__file__).parent
    # Go up to package root and into tests/data/
    snapshot_path = package_dir.parent.parent / 'tests' / 'data' / 'test_data.npz'
    return str(snapshot_path)


def load_snapshot(filename: Optional[str] = None) -> KFunctionsSnapshot:
    """Load complete k-functions snapshot from .npz file.

    Args:
        filename: Path to .npz file containing test data.
                  If None, uses default test_data.npz file.

    Returns:
        KFunctionsSnapshot object with all loaded data
    """
    if filename is None:
        filename = get_default_snapshot_path()
    data = np.load(filename)

    # Extract scalar parameters (handle 0-d arrays)
    def get_scalar(key: str) -> float:
        val = data[key]
        return float(val) if hasattr(val, 'shape') and val.shape == () else val

    # Load all input parameters
    f0 = get_scalar('f0')
    # Note: .npz stores ApOverf0 = KAp_LCDM / f0, so we multiply back to get KAp_LCDM
    params = InputParams(
        f0=f0,
        kmin=get_scalar('kmin'),
        kmax=get_scalar('kmax'),
        Nk=int(get_scalar('Nk')),
        nquadSteps=int(get_scalar('nquadSteps')),
        NQ=int(get_scalar('NQ')),
        NR=int(get_scalar('NR')),
        KA_LCDM=get_scalar('A'),
        KAp_LCDM=get_scalar('ApOverf0') * f0,
        KR1_LCDM=get_scalar('CFD3'),
        KR1p_LCDM=get_scalar('CFD3p'),
        sigma2v=get_scalar('sigma2v')
    )

    # Input arrays
    arrays = InputArrays(
        k_in=data['k_in'],
        Pk_in=data['P_wiggle'],
        Pk_nw_in=data['P_nowiggle'],
        f_in=data['f']
    )

    # Validate input arrays
    # Check all arrays are 1D
    for field_name in ['k_in', 'Pk_in', 'Pk_nw_in', 'f_in']:
        arr = getattr(arrays, field_name)
        if arr.ndim != 1:
            raise ValueError(f"Input array '{field_name}' must be 1D, got shape {arr.shape}")

    # Check all arrays have the same size
    sizes = {
        'k_in': len(arrays.k_in),
        'Pk_in': len(arrays.Pk_in),
        'Pk_nw_in': len(arrays.Pk_nw_in),
        'f_in': len(arrays.f_in)
    }
    if len(set(sizes.values())) != 1:
        raise ValueError(f"All input arrays must have the same size, got {sizes}")

    # Check k_in is strictly increasing
    if not np.all(np.diff(arrays.k_in) > 0):
        raise ValueError("k_in must be strictly increasing")

    # Check k_in values are positive
    if not np.all(arrays.k_in > 0):
        raise ValueError("k_in values must be positive")

    # Expected k-functions outputs
    def load_kfunctions(prefix: str) -> KFunctions:
        # Compute output k-grid from parameters
        k_out = np.geomspace(params.kmin, params.kmax, params.Nk).astype(np.float64)

        return KFunctions(
            k=k_out,
            P22dd=data[f'{prefix}_P22dd'],
            P22du=data[f'{prefix}_P22du'],
            P22uu=data[f'{prefix}_P22uu'],
            P13dd=data[f'{prefix}_P13dd'],
            P13du=data[f'{prefix}_P13du'],
            P13uu=data[f'{prefix}_P13uu'],
            I1udd1A=data[f'{prefix}_I1udd1A'],
            I2uud1A=data[f'{prefix}_I2uud1A'],
            I2uud2A=data[f'{prefix}_I2uud2A'],
            I3uuu2A=data[f'{prefix}_I3uuu2A'],
            I3uuu3A=data[f'{prefix}_I3uuu3A'],
            I2uudd1BpC=data[f'{prefix}_I2uudd1BpC'],
            I2uudd2BpC=data[f'{prefix}_I2uudd2BpC'],
            I3uuud2BpC=data[f'{prefix}_I3uuud2BpC'],
            I3uuud3BpC=data[f'{prefix}_I3uuud3BpC'],
            I4uuuu2BpC=data[f'{prefix}_I4uuuu2BpC'],
            I4uuuu3BpC=data[f'{prefix}_I4uuuu3BpC'],
            I4uuuu4BpC=data[f'{prefix}_I4uuuu4BpC'],
            Pb1b2=data[f'{prefix}_Pb1b2'],
            Pb1bs2=data[f'{prefix}_Pb1bs2'],
            Pb22=data[f'{prefix}_Pb22'],
            Pb2s2=data[f'{prefix}_Pb2s2'],
            Ps22=data[f'{prefix}_Ps22'],
            Pb2theta=data[f'{prefix}_Pb2theta'],
            Pbs2theta=data[f'{prefix}_Pbs2theta'],
            sigma32PSL=data[f'{prefix}_sigma32PSL'],
            pkl=data[f'{prefix}_pkl']
        )

    kfuncs_wiggle = load_kfunctions('expected_wiggle')
    kfuncs_nowiggle = load_kfunctions('expected_nowiggle')

    return KFunctionsSnapshot(
        params=params,
        arrays=arrays,
        kfuncs_wiggle=kfuncs_wiggle,
        kfuncs_nowiggle=kfuncs_nowiggle
    )
