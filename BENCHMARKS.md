# Benchmarks

This document tracks benchmark results for `fkptjax` across different machines and computation backends (NumPy, JAX CPU, JAX GPU).

We report three timings for each backend:

- **Initialize** — time to construct and initialize the calculator
- **First eval** — first call to `evaluate()` (includes JAX compilation overhead)
- **Avg eval** — average evaluation time over 100 calls

**Benchmark procedure:** The `fkptjax` timings below are obtained by running the benchmark script in the repository:

```bash
python tests/test.py
```

We also report the **full k-loop time** from running:

```bash
./fkpt chatty=1 Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 fnamePS=pkl_z05.dat
```

**NOTE:** The `./fkpt ...` timing refers to the reference C implementation from the original `fkpt` codebase, available at https://github.com/alejandroaviles/fkpt. This serves as a baseline for comparing `fkptjax` performance against the established C version.



| Platform            | NumPy Init (ms) | NumPy First Eval (ms) | NumPy Avg Eval (ms) | JAX CPU Init (ms) | JAX CPU First Eval (ms) | JAX CPU Avg Eval (ms) | JAX GPU Init (ms) | JAX GPU First Eval (ms) | JAX GPU Avg Eval (ms) | FKPT Total (ms) |
| ------------------- | --------------: | --------------------: | ------------------: | ------------: | ------------------: | ----------------: | ------------: | ------------------: | --------------:   | --------------: |
| Perlmutter (A100)   | 32.91           | 127.16                | 118.1               | 2249.65       | 853.37              | 31.2              | 3211.55       | 1762.78             | 18.7              | 249.12          |
| Entropy (RTX 3090)  | 35.36           | 151.01                | 138.5               | 1355.23       | 796.71              | 52.0              | 3274.40       | 1859.43             | 15.4              | 229.00          |
| Google Colab (T4)   | 52.15           | 270.79                | 267.8               | 2880.20       | 5536.64             | 281.5             | 4098.79       | 4006.16             | 26.2              | 407.97          |
| Apple 1 Max (CPU)   | 17.25           | 115.04                | 108.0               | 1720.71       | 719.09              | 23.8              | N/A           | N/A                 | N/A               | 128.5           |
