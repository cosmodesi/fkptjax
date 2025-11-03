import numpy as np

import matplotlib.pyplot as plt


def plot_input_arrays(k_in, Pk_in, Pk_nw_in, f_in, f0):

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    delta = Pk_in - Pk_nw_in
    ax[0].loglog(k_in, Pk_nw_in, lw=6, alpha=0.3, label='No-Wiggle')
    ax[0].loglog(k_in, Pk_in, lw=1, label='+ Wiggles')
    ax[0].loglog(k_in, Pk_nw_in + 10 * delta, lw=1, label='+ 10 x Wiggles')
    ax[0].set_xlabel('k [h/Mpc]')
    ax[0].set_ylabel('Input linear matter power P(k) [(Mpc/h)$^3$]')
    ax[0].legend()

    ax[1].semilogx(k_in, f_in)
    ax[1].axhline(f0, ls='--', label='f0')
    ax[1].legend()
    ax[1].set_xlabel('k [h/Mpc]')
    ax[1].set_ylabel('Input scale-dependent growth rate f(k)')

    plt.tight_layout()
    return fig, ax


def plot_one_loop(k_out, P22dd, P22du, P22uu,
                  P13dd, P13du, P13uu,
                  I1udd1A, I2uud1A, I2uud2A,
                  I3uuu2A, I3uuu3A,
                  I2uudd1BpC, I2uudd2BpC,
                  I3uuud2BpC, I3uuud3BpC,
                  I4uuuu2BpC, I4uuuu3BpC, I4uuuu4BpC):
    """Plot 1-loop k-functions."""

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    for i, label in enumerate(('no-wiggles', 'wiggles')):
        ls = '-' if i == 0 else '--'
        ax[0,0].loglog(k_out, P22dd[i], c='C0', ls=ls, label=f'P22dd' if i == 0 else None)
        ax[0,0].loglog(k_out, P22du[i], c='C1', ls=ls, label=f'P22du' if i == 0 else None)
        ax[0,0].loglog(k_out, P22uu[i], c='C2', ls=ls, label=f'P22uu' if i == 0 else None)
        ax[0,1].loglog(k_out, -P13dd[i], c='C0', ls=ls, label=f'-P13dd' if i == 0 else None)
        ax[0,1].loglog(k_out, -P13du[i], c='C1', ls=ls, label=f'-P13du' if i == 0 else None)
        ax[0,1].loglog(k_out, -P13uu[i], c='C2', ls=ls, label=f'-P13uu' if i == 0 else None)
        ax[1,0].loglog(k_out, np.abs(I1udd1A[i]), c='C0', ls=ls, label=f'|I1udd1A|' if i == 0 else None)
        ax[1,0].loglog(k_out, np.abs(I2uud1A[i]), c='C1', ls=ls, label=f'|I2uud1A|' if i == 0 else None)
        ax[1,0].loglog(k_out, np.abs(I2uud2A[i]), c='C2', ls=ls, label=f'|I2uud2A|' if i == 0 else None)
        ax[1,0].loglog(k_out, np.abs(I3uuu2A[i]), c='C3', ls=ls, label=f'|I3uuu2A|' if i == 0 else None)
        ax[1,0].loglog(k_out, np.abs(I3uuu3A[i]), c='C4', ls=ls, label=f'|I3uuu3A|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I2uudd1BpC[i]), c='C0', ls=ls, label=f'|I2uudd1BpC|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I2uudd2BpC[i]), c='C1', ls=ls, label=f'|I2uudd2BpC|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I3uuud2BpC[i]), c='C2', ls=ls, label=f'|I3uuud2BpC|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I3uuud3BpC[i]), c='C3', ls=ls, label=f'|I3uuud3BpC|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I4uuuu2BpC[i]), c='C4', ls=ls, label=f'|I4uuuu2BpC|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I4uuuu3BpC[i]), c='C5', ls=ls, label=f'|I4uuuu3BpC|' if i == 0 else None)
        ax[1,1].loglog(k_out, np.abs(I4uuuu4BpC[i]), c='C6', ls=ls, label=f'|I4uuuu4BpC|' if i == 0 else None)
    for axi in ax.flatten():
        axi.loglog([], [], 'k--', label='No Wiggles')
        axi.set_xlabel('k [h/Mpc]')
        axi.legend()
    ax[0,0].set_ylabel('1-loop P22(k) [(Mpc/h)$^3$]')
    ax[0,1].set_ylabel('1-loop -P13(k) [(Mpc/h)$^3$]')
    ax[1,0].set_ylabel('1-loop TNS A(k) Functions [(Mpc/h)$^3$]')
    ax[1,1].set_ylabel('1-loop TNS D(k) = B+C-G Functions [(Mpc/h)$^3$]')

    plt.tight_layout()
    return fig, ax


def plot_bias_terms(k_out, Pb1b2, Pb1bs2, Pb22, Pb2s2, Ps22, Pb2theta, Pbs2theta):
    """Plot bias term k-functions."""

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i, label in enumerate(('no-wiggles', 'wiggles')):
        ls = '-' if i == 0 else '--'
        ax[0].plot(k_out, Pb1b2[i], c='C0', ls=ls, label='$b_1\\times b_2$ Pb1b2' if i == 0 else None)
        ax[0].plot(k_out, Pb1bs2[i], c='C1', ls=ls, label='$b_1\\times b_{s^2}$ Pb1bs2' if i == 0 else None)
        ax[1].plot(k_out, Pb22[i], c='C2', ls=ls, label='$b_2\\times b_2$ Pb22' if i == 0 else None)
        ax[1].plot(k_out, Pb2s2[i], c='C3', ls=ls, label='$b_2\\times b_{s^2}$ Pb2s2' if i == 0 else None)
        ax[1].plot(k_out, Ps22[i], c='C4', ls=ls, label='$b_{s^2}\\times b_{s^2}$ Ps22' if i == 0 else None)
        ax[0].plot(k_out, Pb2theta[i], c='C5', ls=ls, label='$b_2\\times \\theta$ Pb2theta' if i == 0 else None)
        ax[0].plot(k_out, Pbs2theta[i], c='C6', ls=ls, label='$b_2\\times b_{s^2}$ Pbs2theta' if i == 0 else None)

    for axi in ax:
        axi.plot([], [], 'k--', label='No Wiggles')
        axi.set_xscale('log')
        axi.set_xlabel('k [h/Mpc]')
        axi.set_ylabel('Bias Terms P_xy(k) [(Mpc/h)$^3$]')
        axi.legend()

    plt.tight_layout()
    return fig, ax
