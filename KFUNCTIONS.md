# Explanation of K-Function Array Names

For details see the [paper](https://doi.org/10.1088/1475-7516/2024/03/049): Mario A. Rodriguez-Meza et al, "fkPT: constraining scale-dependent modified gravity with the full-shape galaxy power spectrum", JCAP03(2024)049.

## P Functions

`P22dd`, `P22du`, `P22uu`, `P13dd`, `P13du`, `P13uu` are the one-loop **P22/P13** pieces for density/velocity combinations, eqs. (4.3)–(4.4) of the paper.  The letters `u` and `d` here refer to the density and velocity (divergence) fields.

## I Functions

The general naming scheme is `I{m}{letters}{i}{S}` with:

* **`I`** — stands for one of the rotationally–invariant **k-functions** (loop integrals) that appear in the power-spectrum expansion $P(k,\mu)=\sum_m\sum_n \mu^{2n},f_0^{,m},I_{mn}(k)$. The code stores many specific $I$-type integrals that make up these $I_{mn}$.
* **`{m}` (the first digit)** — **power of the growth-rate factor** $f_0$, i.e., how many velocity (u) factors the term carries. Example: `I4…` multiplies $f_0^4$; `I2…` multiplies $f_0^2$. This matches the paper’s $f_0^m$ bookkeeping.
* **`{letters}`** — which **fields** appear in the underlying correlator:

  * `d` = density field $\delta$
  * `u` = velocity(-divergence) field used in TNS RSD $u / \vartheta$; see how $A_{\mathrm{TNS}}$ and $D$ use mixed $\delta$–velocity correlators. So `uuud` means three velocities and one density; `uudd` = two of each, etc.
* **`{i}` (the small number after the letters)** — an **enumerator** distinguishing different angular/kernel structures that occur with the same field content (different basis integrals after the μ-algebra).
* **`{S}` (the suffix)** — **which RSD building block** the integral belongs to:

  * `A` → contributes to the **Taruya–Nishimichi–Saito** **A** function $A_{\mathrm{TNS}}$ (bispectrum-type term).
  * `BpC` → contributes to the **D** function’s **B + C - G combination** where the “− G” subtraction is the Gaussian-damping counterterm the code removes explicitly, governed by the `sigma2v` parameter.

## Bias Functions

These have names like `Pb1b2` and follow the bias operator naming scheme used in standard LSS EFT codes and the bias literature (McDonald 2006, Desjacques et al 2018 review).

The leading `P` means that these are power spectrum contributions. The rest of the name identifies the two fields being correlated:

| substring   | meaning                                             |
| ----------- | --------------------------------------------------- |
| `b1`        | linear bias operator (coefficient $b_1$)            |
| `b2`        | quadratic (local) bias operator (coefficient $b_2$) |
| `s2`        | tidal shear operator (s^2) (coefficient $b_{s²}$)   |
| `theta`     | velocity divergence field                           |
