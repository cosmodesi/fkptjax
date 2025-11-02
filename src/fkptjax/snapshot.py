from dataclasses import dataclass

import numpy as np

from fkptjax.types import Float64NDArray


@dataclass
class CosmologyParams:
    """Cosmological parameters from test data"""
    f0: float  # Growth rate at k→0

@dataclass
class KGridParams:
    """Output k-grid parameters"""
    kmin: float
    kmax: float
    Nk: int

@dataclass
class NumericalParams:
    """Numerical integration parameters"""
    nquadSteps: int  # Number of quadrature steps for k-integration
    NQ: int          # Number of Gauss-Legendre points for Q-functions
    NR: int          # Number of Gauss-Legendre points for R-functions

@dataclass
class KernelConstants:
    """SPT kernel constants"""
    KA_LCDM: float   # Kernel constant A
    KAp_LCDM: float  # Kernel constant Ap
    KR1_LCDM: float  # Kernel constant CFD3
    KR1p_LCDM: float # Kernel constant CFD3'

@dataclass
class SigmaValues:
    """Variance and damping integrals"""
    sigma2v: float  # Velocity dispersion σ²_v

@dataclass
class LinearPowerSpectrum:
    """Input linear power spectrum"""
    k: Float64NDArray  # k values [h/Mpc]
    P: Float64NDArray  # P(k) [(Mpc/h)³]
    f: Float64NDArray  # f(k) growth rate

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
    cosmology: CosmologyParams
    k_grid: KGridParams
    numerical: NumericalParams
    kernels: KernelConstants
    sigma_values: SigmaValues
    # Inputs
    ps_wiggle: LinearPowerSpectrum
    ps_nowiggle: LinearPowerSpectrum
    # Expected outputs
    kfuncs_wiggle: KFunctions
    kfuncs_nowiggle: KFunctions


def load_snapshot(filename: str) -> KFunctionsSnapshot:
    """Load complete k-functions snapshot from .npz file.

    Args:
        filename: Path to .npz file containing test data

    Returns:
        KFunctionsSnapshot object with all loaded data
    """
    data = np.load(filename)

    # Extract scalar parameters (handle 0-d arrays)
    def get_scalar(key: str) -> float:
        val = data[key]
        return float(val) if hasattr(val, 'shape') and val.shape == () else val

    # Cosmology parameters
    f0 = get_scalar('f0')
    cosmology = CosmologyParams(f0=f0)

    # K-grid parameters
    k_grid = KGridParams(
        kmin=get_scalar('kmin'),
        kmax=get_scalar('kmax'),
        Nk=int(get_scalar('Nk'))
    )

    # Numerical parameters
    numerical = NumericalParams(
        nquadSteps=int(get_scalar('nquadSteps')),
        NQ=int(get_scalar('NQ')),
        NR=int(get_scalar('NR'))
    )

    # Kernel constants
    # Note: .npz stores ApOverf0 = KAp_LCDM / f0, so we multiply back to get KAp_LCDM
    kernels = KernelConstants(
        KA_LCDM=get_scalar('A'),
        KAp_LCDM=get_scalar('ApOverf0') * f0,
        KR1_LCDM=get_scalar('CFD3'),
        KR1p_LCDM=get_scalar('CFD3p')
    )

    # Sigma values
    sigma_values = SigmaValues(
        sigma2v=get_scalar('sigma2v')
    )

    # Input power spectra
    # Note: f is shared between wiggle and no-wiggle
    k_in = data['k_in']
    f_in = data['f']

    ps_wiggle = LinearPowerSpectrum(
        k=k_in,
        P=data['P_wiggle'],
        f=f_in
    )

    ps_nowiggle = LinearPowerSpectrum(
        k=k_in,
        P=data['P_nowiggle'],
        f=f_in
    )

    # Expected k-functions outputs
    def load_kfunctions(prefix: str) -> KFunctions:
        # Compute output k-grid from parameters
        k_out = np.geomspace(k_grid.kmin, k_grid.kmax, k_grid.Nk)

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
        cosmology=cosmology,
        k_grid=k_grid,
        numerical=numerical,
        kernels=kernels,
        sigma_values=sigma_values,
        ps_wiggle=ps_wiggle,
        ps_nowiggle=ps_nowiggle,
        kfuncs_wiggle=kfuncs_wiggle,
        kfuncs_nowiggle=kfuncs_nowiggle
    )
