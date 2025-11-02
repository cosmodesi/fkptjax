# Test Data Format

This directory contains reference data for testing and validating the k-functions calculators.

## File Format

Test data is stored in `test_data.npz`, a NumPy compressed archive containing all inputs, parameters, and expected outputs needed for calculator testing and validation.

## Loading Test Data

Use the `load_snapshot()` function from `fkptjax.snapshot`:

```python
from fkptjax.snapshot import load_snapshot

snapshot = load_snapshot('tests/data/test_data.npz')

# Access input data
k_in = snapshot.ps_wiggle.k
P_wiggle = snapshot.ps_wiggle.P
P_nowiggle = snapshot.ps_nowiggle.P
f = snapshot.ps_wiggle.f

# Access configuration
kmin = snapshot.k_grid.kmin
Nk = snapshot.k_grid.Nk
nquadSteps = snapshot.numerical.nquadSteps
NQ = snapshot.numerical.NQ
NR = snapshot.numerical.NR
sigma2v = snapshot.sigma_values.sigma2v

# Access kernel constants
A = snapshot.kernels.KA_LCDM
ApOverf0 = snapshot.kernels.KAp_LCDM / snapshot.cosmology.f0

# Access expected outputs for validation
expected_wiggle = snapshot.kfuncs_wiggle
expected_nowiggle = snapshot.kfuncs_nowiggle
```

The `load_snapshot()` function returns a `KFunctionsSnapshot` object that provides a consistent interface to the test data.

## Required Arrays and Parameters

### Input Arrays

These arrays provide the input linear power spectra and growth rates:

| Key | Shape | Description |
|-----|-------|-------------|
| `k_in` | `(n_input,)` | Input wavenumber grid [h/Mpc] |
| `P_wiggle` | `(n_input,)` | Linear power spectrum with BAO wiggles [(Mpc/h)³] |
| `P_nowiggle` | `(n_input,)` | No-wiggle linear power spectrum [(Mpc/h)³] |
| `f` | `(n_input,)` | Growth rate f(k) (same for wiggle and no-wiggle) |

**Note**: All input arrays must have the same length `n_input`.

### Configuration Parameters

Scalar parameters that control the calculation grid and cosmology:

| Key | Type | Description |
|-----|------|-------------|
| `kmin` | float | Minimum output k value [h/Mpc] |
| `kmax` | float | Maximum output k value [h/Mpc] |
| `Nk` | int | Number of output k bins |
| `nquadSteps` | int | Number of quadrature steps for k-integration |
| `NQ` | int | Number of Gauss-Legendre points for Q-functions (test data uses 10) |
| `NR` | int | Number of Gauss-Legendre points for R-functions (test data uses 10) |
| `sigma2v` | float | Velocity dispersion σ²_v [(Mpc/h)²] |
| `f0` | float | Growth rate at z=0 (normalization constant) |
| `A` | float | Kernel constant A (test data uses 1.0) |
| `ApOverf0` | float | Kernel constant A'/f₀ (test data uses 0.0) |
| `CFD3` | float | Kernel constant CFD3 (test data uses 1.0) |
| `CFD3p` | float | Kernel constant CFD3' (test data uses 1.0) |

**Note**: The expected outputs were computed with the kernel constants shown above. When creating new test data with different kernel constants, the expected outputs must be recomputed accordingly.

### Expected Output Arrays - Wiggle

Each array has shape `(Nk,)` containing expected k-function values on the output grid:

#### P22 Components (1-loop density-density power)
- `expected_wiggle_P22dd` - δ-δ component
- `expected_wiggle_P22du` - δ-θ component
- `expected_wiggle_P22uu` - θ-θ component

#### P13 Components (1-loop matter power with tree-level coupling)
- `expected_wiggle_P13dd` - δ-δ component
- `expected_wiggle_P13du` - δ-θ component
- `expected_wiggle_P13uu` - θ-θ component

#### RSD A-Terms (Redshift-space distortion A-terms)
- `expected_wiggle_I1udd1A` - μ² term in RSD expansion
- `expected_wiggle_I2uud1A` - μ² term
- `expected_wiggle_I2uud2A` - μ² term
- `expected_wiggle_I3uuu2A` - μ⁴ term
- `expected_wiggle_I3uuu3A` - μ⁴ term

#### RSD D-Terms (B+C-G terms with velocity damping)
- `expected_wiggle_I2uudd1BpC` - μ² term with damping
- `expected_wiggle_I2uudd2BpC` - μ² term
- `expected_wiggle_I3uuud2BpC` - μ⁴ term with damping
- `expected_wiggle_I3uuud3BpC` - μ⁴ term
- `expected_wiggle_I4uuuu2BpC` - μ⁶ term with damping
- `expected_wiggle_I4uuuu3BpC` - μ⁶ term
- `expected_wiggle_I4uuuu4BpC` - μ⁶ term

#### Bias Terms (Galaxy bias corrections)
- `expected_wiggle_Pb1b2` - b₁b₂ cross term
- `expected_wiggle_Pb1bs2` - b₁bs₂ cross term
- `expected_wiggle_Pb22` - b₂² term
- `expected_wiggle_Pb2s2` - b₂s₂ cross term
- `expected_wiggle_Ps22` - s₂² term
- `expected_wiggle_Pb2theta` - b₂θ cross term
- `expected_wiggle_Pbs2theta` - bs₂θ cross term

#### Additional Terms
- `expected_wiggle_sigma32PSL` - Velocity dispersion integral
- `expected_wiggle_pkl` - Linear P(k) interpolated to output grid [(Mpc/h)³]

### Expected Output Arrays - No-Wiggle

Identical structure to wiggle arrays, but computed from the no-wiggle power spectrum:

All arrays follow the naming pattern: `expected_nowiggle_<field_name>` where `<field_name>` matches the wiggle array names (e.g., `expected_nowiggle_P22dd`, `expected_nowiggle_I1udd1A`, etc.).

**Total**: 27 arrays for wiggle + 27 arrays for no-wiggle = **54 expected output arrays**

### Summary

The complete `.npz` file contains:
- 4 input arrays (`k_in`, `P_wiggle`, `P_nowiggle`, `f`)
- 12 configuration parameters (6 grid/numerical + 4 kernel constants + 2 physics)
- 54 expected output arrays (27 wiggle + 27 no-wiggle)

**Total: 70 arrays**

## Data Usage in Tests

The test workflow (`tests/test.py`) uses this data as follows:

1. **Load Data**: Read `.npz` file to extract inputs, parameters, and expected outputs
2. **Setup**: Call `setup_kfunctions()` with inputs and parameters to initialize grids
3. **Initialize**: Create calculator instance and call `initialize()` with grid data
4. **Evaluate**: Run calculator with input power spectra, growth rates, and kernel constants
5. **Validate**: Compare computed outputs against expected outputs using `validate_kfunctions()`
   - Tolerance: `rtol=1e-5, atol=1e-8`
   - All 27 k-function fields are validated for both wiggle and no-wiggle

## Creating Test Data

To generate a new `.npz` file from reference calculations:

```python
import numpy as np

# Collect all required data
data = {
    # Input arrays
    'k_in': k_array,
    'P_wiggle': P_wiggle_array,
    'P_nowiggle': P_nowiggle_array,
    'f': f_array,

    # Configuration parameters
    'kmin': kmin_value,
    'kmax': kmax_value,
    'Nk': Nk_value,
    'nquadSteps': nquadSteps_value,
    'NQ': NQ_value,
    'NR': NR_value,
    'sigma2v': sigma2v_value,
    'f0': f0_value,

    # Kernel constants
    'A': A_value,
    'ApOverf0': ApOverf0_value,
    'CFD3': CFD3_value,
    'CFD3p': CFD3p_value,

    # Expected wiggle outputs (27 arrays)
    'expected_wiggle_P22dd': ...,
    'expected_wiggle_P22du': ...,
    # ... (all 27 wiggle fields)

    # Expected no-wiggle outputs (27 arrays)
    'expected_nowiggle_P22dd': ...,
    'expected_nowiggle_P22du': ...,
    # ... (all 27 no-wiggle fields)
}

# Save to compressed .npz format
np.savez_compressed('test_data.npz', **data)
```

## Validation Tolerance

The validation uses NumPy's `allclose()` with:
- **Relative tolerance**: `rtol = 1e-5` (0.001%)
- **Absolute tolerance**: `atol = 1e-8`

An element passes if: `|computed - expected| <= atol + rtol * |expected|`

## Notes on Stored vs Computed Values

The `.npz` file contains only the minimal data required for testing:

- **Stored**: Input power spectra, grid parameters, kernel constants, and expected outputs
- **Not stored (computed)**: Output k-grid (reconstructed from kmin/kmax/Nk), quadrature grids (generated from scipy)
- **Placeholders**: Some fields in `KFunctionsSnapshot` are set to placeholder values (e.g., cosmology.Om=0.0) since they are not needed for validation
