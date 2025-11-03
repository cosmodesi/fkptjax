# fkptjax

Perturbation theory calculations for LCDM and Modified Gravity theories using "fk"-Kernels implemented in Python with JAX.

Based on the C code at https://github.com/alejandroaviles/fkpt, which is based on the paper Rodriguez-Meza, M. A. et al, "fkPT: Constraining scale-dependent modified gravity with the full-shape galaxy power spectrum", [JCAP03(2024)049](https://doi.org/10.1088/1475-7516/2024/03/049).

## Installation

Install from source:

```bash
pip install .
```

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python 3.10+
- JAX 0.4.0+
- NumPy 1.24.0+
- SciPy 1.10.0+

## Usage

```python
import fkptjax

# Example usage will be added as the package develops
```

See the `examples/` directory for more detailed usage examples.

## Development

Run tests:

```bash
pytest
```

Run benchmarks (default: all calculators, 100 runs):

```bash
python tests/test.py
```

Run benchmarks with options:

```bash
# Run only NumPy calculator
python tests/test.py -c numpy

# Run with custom data file and 50 runs
python tests/test.py -d mydata.npz -n 50

# Run both NumPy and JAX CPU calculators
python tests/test.py -c numpy jax-cpu

# Show all options
python tests/test.py --help
```

Run tests with coverage:

```bash
pytest --cov=fkptjax
```

Type checking:

```bash
mypy src/fkptjax
```

## License

MIT License - see LICENSE file for details.

## Authors

- David Kirkby <dkirkby@uci.edu>
- Matthew Dowicz <mdowicz@uci.edu>
