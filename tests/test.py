import argparse
from pathlib import Path
import sys
import jax

from fkptjax.snapshot import load_snapshot, get_default_snapshot_path
from fkptjax.util import measure_kfunctions
from fkptjax.calculate_numpy import NumpyCalculator
from fkptjax.calculate_jax import JaxCalculator


def check_gpu_available():
    """Check if GPU is available for JAX."""
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        return len(gpu_devices) > 0, devices
    except Exception:
        return False, []


def run_benchmarks(data_file=None, nruns=100, calculators=None):
    """Run calculator benchmarks.

    Args:
        data_file: Path to .npz test data file (default: tests/data/test_data.npz)
        nruns: Number of benchmark runs (default: 100)
        calculators: List of calculators to test (default: ['all'])

    Returns:
        True if all validations passed, False otherwise
    """
    # Determine data file path
    if data_file is None:
        data_file = get_default_snapshot_path()

    data_file = Path(data_file)
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}", file=sys.stderr)
        return False

    # Determine which calculators to run
    if calculators is None:
        calculators = ['all']
    if 'all' in calculators:
        calculators = ['numpy', 'jax-cpu', 'jax-gpu']

    # Load snapshot
    print(f"Loading snapshot from: {data_file}")
    snapshot = load_snapshot(str(data_file))

    # Check GPU availability
    has_gpu, devices = check_gpu_available()

    all_passed = True

    # Run NumPy calculator
    if 'numpy' in calculators:
        print(f"\nMeasuring k-functions using NumPy calculator:")
        passed = measure_kfunctions(NumpyCalculator, snapshot, nruns=nruns)
        all_passed = all_passed and passed

    # Run JAX CPU calculator
    if 'jax-cpu' in calculators:
        print(f"\nMeasuring k-functions using JAX calculator (CPU):")
        jax.config.update('jax_default_device', jax.devices('cpu')[0])
        passed = measure_kfunctions(JaxCalculator, snapshot, nruns=nruns)
        all_passed = all_passed and passed

    # Run JAX GPU calculator
    if 'jax-gpu' in calculators:
        if has_gpu:
            print(f"\nMeasuring k-functions using JAX calculator (GPU):")
            jax.config.update('jax_default_device', jax.devices('gpu')[0])
            passed = measure_kfunctions(JaxCalculator, snapshot, nruns=nruns)
            all_passed = all_passed and passed
        else:
            print(f"\nGPU requested but not available. Available devices: {devices}")
            print("Skipping JAX GPU benchmark.")
            # Don't fail if GPU was requested but not available

    return all_passed


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Run k-functions calculator tests and benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with default settings
  python tests/test.py

  # Run only NumPy calculator with custom data file
  python tests/test.py -d mydata.npz -c numpy

  # Run JAX CPU tests with 50 runs
  python tests/test.py -c jax-cpu -n 50

  # Run NumPy and JAX CPU tests
  python tests/test.py -c numpy jax-cpu
        """
    )

    parser.add_argument(
        '-d', '--data',
        type=str,
        default=None,
        help='Path to .npz test data file (default: tests/data/test_data.npz)'
    )

    parser.add_argument(
        '-n', '--nruns',
        type=int,
        default=100,
        help='Number of benchmark runs (default: 100)'
    )

    parser.add_argument(
        '-c', '--calculators',
        nargs='+',
        choices=['numpy', 'jax-cpu', 'jax-gpu', 'all'],
        default=['all'],
        help='Which calculators to test (default: all)'
    )

    args = parser.parse_args()

    success = run_benchmarks(
        data_file=args.data,
        nruns=args.nruns,
        calculators=args.calculators
    )

    if not success:
        sys.exit(1)


# Pytest test functions
def test_numpy():
    """Test NumPy calculator with pytest."""
    assert run_benchmarks(nruns=10, calculators=['numpy'])


def test_jax_cpu():
    """Test JAX CPU calculator with pytest."""
    assert run_benchmarks(nruns=10, calculators=['jax-cpu'])


def test_jax_gpu():
    """Test JAX GPU calculator with pytest (skips if no GPU)."""
    # This will skip gracefully if GPU is not available
    assert run_benchmarks(nruns=10, calculators=['jax-gpu'])


if __name__ == "__main__":
    main()
