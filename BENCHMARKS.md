# Benchmarks

This document tracks benchmark results for `fkptjax` across different machines and computation backends (NumPy, JAX CPU, JAX GPU).

We report three timings for each backend:

- **Initialize** — time to construct and initialize the calculator
- **First eval** — first call to `evaluate()` (includes JAX compilation overhead)
- **Avg eval** — average evaluation time over 100 calls

We also report the **full k-loop time** from running:

```bash
./fkpt chatty=1 Om=0.3 h=0.7 model=HS fR0=1.0e-6 suffix=_test zout=0.5 fnamePS=pkl_z05.dat
```

**NOTE:** The `./fkpt ...` timing refers to the reference C implementation from the original `fkpt` codebase, available at https://github.com/alejandroaviles/fkpt. This serves as a baseline for comparing `fkptjax` performance against the established C version.

| Platform            | NumPy Init (ms) | NumPy First Eval (ms) | NumPy Avg Eval (ms) | JAX CPU Init (ms) | JAX CPU First Eval (ms) | JAX CPU Avg Eval (ms) | JAX GPU Init (ms) | JAX GPU First Eval (ms) | JAX GPU Avg Eval (ms) | FKPT Total (ms) |
| ------------------- | --------------: | --------------------: | ------------------: | ------------: | ------------------: | ----------------: | ------------: | ------------------: | --------------:   | --------------: |
| Perlmutter (A100)   | 32.66           | 130.73                | 118.4               | 1992.12       | 882.00              | 32.8              | 2945.05       | 1798.78             | 19.4              | 249.12          |
| Entropy (RTX 3090)  | 34.99           | 152.14                | 136.5               | 1372.46       | 793.30              | 51.5              | 2870.60       | 1874.32             | 17.8              | 229.00          |
| Google Colab (T4)   | 45.61           | 285.79                | 279.5               | 2804.76       | 6509.02             | 300.4             | 4402.99       | 4241.74             | 27.0              | 407.97          |
| Apple 1 Max (CPU)   | 17.25           | 115.04                | 108.0               | 1720.71       | 719.09              | 23.8              | N/A           | N/A                 | N/A               | 128.5           |
