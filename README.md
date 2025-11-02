# fkptjax

Perturbation theory calculations for LCDM and Modified Gravity theories using "fk"-Kernels implemented in Python with JAX.

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
