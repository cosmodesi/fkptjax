import time

import numpy as np

from scipy.special import roots_legendre

from fkptjax.types import Float64NDArray, KFunctionsInitData, KFunctionsOut, KFunctionsCalculator
from fkptjax.snapshot import KFunctionsSnapshot


def setup_kfunctions(
        k_in: Float64NDArray,
        kmin: float, kmax: float, Nk: int,
        nquadSteps: int, NQ: int=10, NR: int=10
        ) -> KFunctionsInitData:

    # Initialize logarithmic output k grid
    logk_grid = np.geomspace(kmin, kmax, Nk).astype(np.float64)

    # Set up quadrature k grid
    pmin = max(k_in[0], 0.01 * kmin)
    pmax = min(k_in[-1], 16.0 * kmax)
    kk_grid = np.geomspace(pmin, pmax, nquadSteps).astype(np.float64)

    # Initialize Gauss-Legendre nodes and weights on [-1,1]
    xxQ, wwQ = roots_legendre(NQ)
    xxR, wwR = roots_legendre(NR)

    return KFunctionsInitData(
        k_in, logk_grid, kk_grid,
        xxQ.astype(np.float64), wwQ.astype(np.float64),
        xxR.astype(np.float64), wwR.astype(np.float64),
    )

def validate_kfunctions(
        X: KFunctionsOut,
        snapshot: KFunctionsSnapshot,
        rtol: float = 1e-5,
        atol: float = 1e-8
        ) -> bool:
    """Validate that the input k-grid matches the snapshot k-grid."""

    ok = True
    for i, name in enumerate(("kfuncs_wiggle", "kfuncs_nowiggle")):
        B = getattr(snapshot, name)
        for field in KFunctionsOut._fields:
            a = getattr(X, field)[i]
            b = getattr(B, field)
            if not np.allclose(a, b, rtol=rtol, atol=atol):
                diff = atol + rtol * np.abs(b)
                nfail = np.where(np.abs(a - b) > diff)[0].size
                max_abs_diff = np.max(np.abs(a - b))
                max_rel_diff = np.max(np.abs(a - b) / (np.abs(b) + atol))
                print(f"{name:<15s}.{field:<10s} validation fails at {nfail:3d}/{len(b)} elements: max diffs {max_abs_diff:<.3e} (abs) {max_rel_diff:<.3e} (rel)")
                ok = False

    return ok

def measure_kfunctions(
        calculator_cls: KFunctionsCalculator,
        snapshot: KFunctionsSnapshot,
        nruns: int = 10
        ) -> bool:
    """Measure k-functions using the provided calculator and snapshot data.

    Returns:
        True if validation passed, False otherwise
    """

    # Prepare k-functions input
    k_in = snapshot.arrays.k_in
    kmin = snapshot.params.kmin
    kmax = snapshot.params.kmax
    Nk = snapshot.params.Nk
    nquadSteps = snapshot.params.nquadSteps
    NQ = snapshot.params.NQ
    NR = snapshot.params.NR

    start_time = time.time()
    kfuncs_in = setup_kfunctions(
        k_in, kmin, kmax, Nk,
        nquadSteps, NQ, NR
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"setup_kfunctions in {1e3 * elapsed_time:.2f} ms")

    start_time = time.time()
    calculator = calculator_cls()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{calculator_cls.__name__}.ctor in {1e3 * elapsed_time:.2f} ms")

    start_time = time.time()
    calculator.initialize(kfuncs_in)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{calculator_cls.__name__}.initialize in {1e3 * elapsed_time:.2f} ms")

    # kernel constants (use values from snapshot)
    A = snapshot.params.KA_LCDM
    ApOverf0 = snapshot.params.KAp_LCDM / snapshot.params.f0
    CFD3 = snapshot.params.KR1_LCDM
    CFD3p = snapshot.params.KR1p_LCDM

    # Calculate k-functions first time to validate results and do any JIT initialization
    Pk_in = snapshot.arrays.Pk_in
    Pk_nw_in = snapshot.arrays.Pk_nw_in
    fk_in = snapshot.arrays.f_in
    sigma2v = snapshot.params.sigma2v
    f0 = snapshot.params.f0
    start_time = time.time()
    kfuncs_out = calculator.evaluate(Pk_in, Pk_nw_in, fk_in, A, ApOverf0, CFD3, CFD3p, sigma2v, f0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"First {calculator_cls.__name__}.evaluate in {1e3 * elapsed_time:.2f} ms")

    # Validate results
    validation_passed = validate_kfunctions(kfuncs_out, snapshot)
    if validation_passed:
        print("K-functions validated successfully against the snapshot!")
    else:
        print("K-functions validation failed!")

    # Measure time for multiple evaluations
    start_time = time.time()
    for _ in range(nruns):
        kfuncs_out = calculator.evaluate(Pk_in, Pk_nw_in, fk_in, A, ApOverf0, CFD3, CFD3p, sigma2v, f0)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Average {calculator_cls.__name__}.evaluate over {nruns} runs: {1e3 * elapsed_time / nruns:.1f} ms")

    return validation_passed
