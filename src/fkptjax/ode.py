import numpy as np

import scipy.integrate


class ModelDerivatives:

    def __init__(self, om, ol, fR0, beta2=1.0/6.0, nHS=1, screening=1, omegaBD=0.0):
        self.invH0 = 2997.92458 # c/H0 in Mpc/h units
        self.fR0 = fR0
        self.om = om
        self.ol = ol
        self.beta2 = beta2
        self.nHS = nHS
        self.screening = screening
        self.omegaBD = omegaBD

    def mass(self, eta):
        return (
            1 / self.invH0 * np.sqrt(1 / (2 * np.abs(self.fR0)))
            * np.pow(self.om * np.exp(-3 * eta) + 4 * self.ol, (2 + self.nHS) / 2)
            / np.pow(self.om + 4 * self.ol, (1 + self.nHS) / 2)
        )

    def mu(self, eta, k):
        k2 = np.square(k)
        return 1 + 2 * self.beta2 * k2 / (k2 + np.exp(2 * eta) * np.square(self.mass(eta)))

    def PiF(self, eta, k):
        return np.square(k) / np.exp(2 * eta) + np.square(self.mass(eta))

    def M2(self, eta):
        return self.screening * (
            9 / (4 * np.square(self.invH0)) * np.square(1 / np.abs(self.fR0))
            * np.pow(self.om * np.exp(-3 * eta) + 4 * self.ol, 5)
            / np.pow(self.om + 4 * self.ol, 4)
        )

    def OmM(self, eta):
        return 1 / (1 + self.ol / self.om * np.exp(3 * eta))

    def H(self, eta):
        return np.sqrt(self.om * np.exp(-3 * eta) + self.ol)

    def f1(self, eta):
        return 3 / (2 * (1 + self.ol / self.om * np.exp(3 * eta)))

    def kpp(self, x, k, p):
        return np.sqrt(np.square(k) + np.square(p) + 2 * k * p * x)

    def A0(self, eta):
        return 1.5 * self.OmM(eta) * np.square(self.H(eta)) / np.square(self.invH0)

    def source_a(self, eta, kf):
        return self.f1(eta) * self.mu(eta, kf)

    def source_b(self, eta, kf, k1, k2):
        return self.f1(eta) * (self.mu(eta, k1) + self.mu(eta, k2) - self.mu(eta, kf))

    def KFL(self, eta, k, k1, k2):
        k2_  = np.square(k)
        k12  = np.square(k1)
        k22  = np.square(k2)
        num  = np.square(k2_ - k12 - k22)
        term0 = 0.5 * num / (k12 * k22) * (self.mu(eta, k1) + self.mu(eta, k2) - 2.0)
        term1 = 0.5 * (k2_ - k12 - k22) / k12 * (self.mu(eta, k1) - 1.0)
        term2 = 0.5 * (k2_ - k12 - k22) / k22 * (self.mu(eta, k2) - 1.0)
        return term0 + term1 + term2

    def source_FL(self, eta, kf, k1, k2):
        return self.f1(eta) * np.square(self.mass(eta)) / self.PiF(eta, kf) * self.KFL(eta, kf, k1, k2)

    def source_dI(self, eta, kf, k1, k2):
        return (
            1/6 * np.sqrt(self.OmM(eta) * self.H(eta) / (np.exp(eta) * self.invH0))
            * np.sqrt(kf) * self.M2(eta) / (self.PiF(eta, kf) * self.PiF(eta, k1) * self.PiF(eta, k2))
        )

    def source_A(self, eta, kf, k1, k2):
        return self.source_a(eta, kf) + self.source_FL(eta, kf, k1, k2) - self.source_dI(eta, kf, k1, k2)

    # Third order helper functions
    def M1(self, eta):
        return 3.0 * np.square(self.mass(eta))

    def M3(self, eta):
        return self.screening * (
            45.0 / (8.0 * np.square(self.invH0)) * np.power(1 / np.abs(self.fR0), 3.0)
            * np.power(self.om * np.exp(-3.0 * eta) + 4.0 * self.ol, 7.0)
            / np.power(self.om + 4.0 * self.ol, 6.0)
        )

    def KFL2(self, eta, x, k, p):
        return (
            2.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0)
            + (p * x / k) * (self.mu(eta, k) - 1.0)
            + (k * x / p) * (self.mu(eta, p) - 1.0)
        )

    def JFL(self, eta, x, k, p):
        return (
            9.0 / (2.0 * self.A0(eta))
            * self.KFL2(eta, x, k, p) * self.PiF(eta, k) * self.PiF(eta, p)
        )

    def D2phiplus(self, eta, x, k, p, Dpk, Dpp, D2f):
        return (
            (1.0 + np.square(x))
            - (2.0 * self.A0(eta) / 3.0)
            * (
                (self.M2(eta) + self.JFL(eta, x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / (3.0 * self.PiF(eta, k) * self.PiF(eta, p))
            )
        ) * Dpk * Dpp + D2f

    def D2phiminus(self, eta, x, k, p, Dpk, Dpp, D2mf):
        return (
            (1.0 + np.square(x))
            - (2.0 * self.A0(eta) / 3.0)
            * (
                (self.M2(eta) + self.JFL(eta, -x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / (3.0 * self.PiF(eta, k) * self.PiF(eta, p))
            )
        ) * Dpk * Dpp + D2mf

    def K3dI(self, eta, x, k, p, Dpk, Dpp, D2f, D2mf):
        kplusp = self.kpp(x, k, p)
        kpluspm = self.kpp(-x, k, p)

        t1 = (
            2.0 * np.square(self.OmM(eta) * self.H(eta) / self.invH0)
            * (self.M2(eta) / (self.PiF(eta, k) * self.PiF(eta, 0)))
        )

        t2 = (
            (1.0 / 3.0) * (np.power(self.OmM(eta), 3.0) * np.power(self.H(eta), 4.0) / np.power(self.invH0, 4))
            * (
                self.M3(eta) - self.M2(eta) * (self.M2(eta) + self.JFL(eta, -1.0, p, p) * (3.0 + 2.0 * self.omegaBD))
                / self.PiF(eta, 0)
            ) / (np.square(self.PiF(eta, p)) * self.PiF(eta, k))
        )

        t3 = (
            np.square(self.OmM(eta) * self.H(eta) / self.invH0)
            * (self.M2(eta) / (self.PiF(eta, p) * self.PiF(eta, kplusp)))
            * (1.0 + np.square(x) + D2f / (Dpk * Dpp))
        )

        t4 = (
            (1.0 / 3.0) * (np.power(self.OmM(eta), 3.0) * np.power(self.H(eta), 4.0) / np.power(self.invH0, 4))
            * (
                self.M3(eta) - self.M2(eta) * (self.M2(eta) + self.JFL(eta, x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / self.PiF(eta, kplusp)
            ) / (np.square(self.PiF(eta, p)) * self.PiF(eta, k))
        )

        t5 = (
            np.square(self.OmM(eta) * self.H(eta) / self.invH0)
            * (self.M2(eta) / (self.PiF(eta, p) * self.PiF(eta, kpluspm)))
            * (1.0 + np.square(x) + D2mf / (Dpk * Dpp))
        )

        t6 = (
            (1.0 / 3.0) * (np.power(self.OmM(eta), 3.0) * np.power(self.H(eta), 4.0) / np.power(self.invH0, 4))
            * (
                self.M3(eta) - self.M2(eta) * (self.M2(eta) + self.JFL(eta, -x, k, p) * (3.0 + 2.0 * self.omegaBD))
                / self.PiF(eta, kpluspm)
            ) / (np.square(self.PiF(eta, p)) * self.PiF(eta, k))
        )

        return t1 + t2 + t3 + t4 + t5 + t6

    def S2a(self, eta, x, k, p):
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * self.mu(eta, kplusp)

    def S2b(self, eta, x, k, p):
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * (self.mu(eta, k) + self.mu(eta, p) - self.mu(eta, kplusp))

    def S2FL(self, eta, x, k, p):
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * (
            self.M1(eta) / (3.0 * self.PiF(eta, kplusp))
            * self.KFL2(eta, x, k, p)
        )

    def S2dI(self, eta, x, k, p):
        kplusp = self.kpp(x, k, p)
        return (
            (1.0 / 6.0) * np.square(self.OmM(eta) * self.H(eta) / (np.exp(eta) * self.invH0))
            * (np.square(kplusp) * self.M2(eta) / (self.PiF(eta, kplusp) * self.PiF(eta, k) * self.PiF(eta, p)))
        )

    def SD2(self, eta, x, k, p):
        return (
            self.S2a(eta, x, k, p) - self.S2b(eta, x, k, p) * np.square(x)
            + self.S2FL(eta, x, k, p) - self.S2dI(eta, x, k, p)
        )

    def S3IIplus(self, eta, x, k, p, Dpk, Dpp, D2f):
        kplusp = self.kpp(x, k, p)
        return (
            -self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kplusp) - 2.0 * self.mu(eta, k))
            * Dpp * (D2f + Dpk * Dpp * np.square(x))
            - self.f1(eta) * (self.mu(eta, kplusp) - self.mu(eta, k)) * Dpk * Dpp * Dpp
            - (
                (self.M1(eta) / (3.0 * self.PiF(eta, kplusp))) * self.f1(eta) * self.KFL2(eta, x, k, p)
                - np.square(self.OmM(eta) * self.H(eta) / self.invH0)
                * (self.M2(eta) * kplusp * kplusp * np.exp(-2.0 * eta))
                / (6.0 * self.PiF(eta, kplusp) * self.PiF(eta, k) * self.PiF(eta, p))
            ) * Dpk * Dpp * Dpp
        )

    def S3IIminus(self, eta, x, k, p, Dpk, Dpp, D2mf):
        kpluspm = self.kpp(-x, k, p)
        return (
            -self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kpluspm) - 2.0 * self.mu(eta, k))
            * Dpp * (D2mf + Dpk * Dpp * np.square(x))
            - self.f1(eta) * (self.mu(eta, kpluspm) - self.mu(eta, k)) * Dpk * Dpp * Dpp
            - (
                (self.M1(eta) / (3.0 * self.PiF(eta, kpluspm))) * self.f1(eta) * self.KFL2(eta, -x, k, p)
                - np.square(self.OmM(eta) * self.H(eta) / self.invH0)
                * (self.M2(eta) * kpluspm * kpluspm * np.exp(-2.0 * eta))
                / (6.0 * self.PiF(eta, kpluspm) * self.PiF(eta, k) * self.PiF(eta, p))
            ) * Dpk * Dpp * Dpp
        )

    def S3FLplus(self, eta, x, k, p, Dpk, Dpp, D2f):
        kplusp = self.kpp(x, k, p)
        return self.f1(eta) * (self.M1(eta) / (3.0 * self.PiF(eta, k))) * (
            (2.0 * np.square(p + k * x) / np.square(kplusp) - 1.0 - (k * x) / p)
            * (self.mu(eta, p) - 1.0) * D2f * Dpp
            + ((np.square(p) + 3.0 * k * p * x + 2.0 * k * k * x * x) / np.square(kplusp))
            * (self.mu(eta, kplusp) - 1.0) * self.D2phiplus(eta, x, k, p, Dpk, Dpp, D2f) * Dpp
            + 3.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0) * Dpk * Dpp * Dpp
        )

    def S3FLminus(self, eta, x, k, p, Dpk, Dpp, D2mf):
        kpluspm = self.kpp(-x, k, p)
        return self.f1(eta) * (self.M1(eta) / (3.0 * self.PiF(eta, k))) * (
            (2.0 * np.square(p - k * x) / np.square(kpluspm) - 1.0 + (k * x) / p)
            * (self.mu(eta, p) - 1.0) * D2mf * Dpp
            + ((np.square(p) - 3.0 * k * p * x + 2.0 * k * k * x * x) / np.square(kpluspm))
            * (self.mu(eta, kpluspm) - 1.0) * self.D2phiminus(eta, x, k, p, Dpk, Dpp, D2mf) * Dpp
            + 3.0 * np.square(x) * (self.mu(eta, k) + self.mu(eta, p) - 2.0) * Dpk * Dpp * Dpp
        )

    # Main third order source functions
    def S3I(self, eta, x, k, p, Dpk, Dpp, D2f, D2mf):
        kplusp = self.kpp(x, k, p)
        kpluspm = self.kpp(-x, k, p)
        return (
            (
                self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kplusp) - self.mu(eta, k)) * D2f * Dpp
                + self.SD2(eta, x, k, p) * Dpk * Dpp * Dpp
            ) * (1.0 - np.square(x)) / (1.0 + np.square(p / k) + 2.0 * (p / k) * x)
            + (
                self.f1(eta) * (self.mu(eta, p) + self.mu(eta, kpluspm) - self.mu(eta, k)) * D2mf * Dpp
                + self.SD2(eta, -x, k, p) * Dpk * Dpp * Dpp
            ) * (1.0 - np.square(x)) / (1.0 + np.square(p / k) - 2.0 * (p / k) * x)
        )

    def S3II(self, eta, x, k, p, Dpk, Dpp, D2f, D2mf):
        return self.S3IIplus(eta, x, k, p, Dpk, Dpp, D2f) + self.S3IIminus(eta, x, k, p, Dpk, Dpp, D2mf)

    def S3FL(self, eta, x, k, p, Dpk, Dpp, D2f, D2mf):
        return self.S3FLplus(eta, x, k, p, Dpk, Dpp, D2f) + self.S3FLminus(eta, x, k, p, Dpk, Dpp, D2mf)

    def S3dI(self, eta, x, k, p, Dpk, Dpp, D2f, D2mf):
        return (
            -(np.square(k) / np.exp(2.0 * eta))
            * (1.0 / (6.0 * self.PiF(eta, k)))
            * self.K3dI(eta, x, k, p, Dpk, Dpp, D2f, D2mf) * Dpk * Dpp * Dpp
        )

    def firstOrder(self, x, y, k):
        f1x = self.f1(x)
        return np.array([y[1], f1x * self.mu(x, k) * y[0] - (2 - f1x) * y[1]])

    def secondOrder(self, x, y, kf, k1, k2):
        f1x = self.f1(x)
        f2x = 2 - f1x
        srcA = self.source_A(x, kf, k1, k2)
        srcB = self.source_b(x, kf, k1, k2)
        return np.array([
            y[1], f1x * self.mu(x, k1) * y[0] - f2x * y[1],
            y[3], f1x * self.mu(x, k2) * y[2] - f2x * y[3],
            y[5], f1x * self.mu(x, kf) * y[4] - f2x * y[5] + srcA * y[0] * y[2],
            y[7], f1x * self.mu(x, kf) * y[6] - f2x * y[7] + srcB * y[0] * y[2]
        ])

    def thirdOrder(self, eta, y, x, k, p):
        f1eta = self.f1(eta)
        f2eta = 2 - f1eta
        kplusp = self.kpp(x, k, p)
        kpluspm = self.kpp(-x, k, p)
        Dpk = y[0]
        Dpp = y[2]
        D2f = y[4]
        D2mf = y[6]
        return np.array([
            y[1], f1eta * self.mu(eta, k) * y[0] - f2eta * y[1],
            y[3], f1eta * self.mu(eta, p) * y[2] - f2eta * y[3],
            y[5], f1eta * self.mu(eta, kplusp) * y[4] - f2eta * y[5] + self.SD2(eta, x, k, p) * y[0] * y[2],
            y[7], f1eta * self.mu(eta, kpluspm) * y[6] - f2eta * y[7] + self.SD2(eta, -x, k, p) * y[0] * y[2],
            y[9], f1eta * self.mu(eta, k) * y[8] - f2eta * y[9]
                + self.S3I(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
                + self.S3II(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
                + self.S3FL(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
                + self.S3dI(eta, x, k, p, Dpk, Dpp, D2f, D2mf)
        ])

class ODESolver:

    def __init__(self, zout, xnow=-4, method='RKQS'):
        self.xstop = np.log(1.0/(1.0+zout))
        self.xnow = xnow
        if method not in ['RKQS', 'scipy_ivp']:
            raise ValueError(f"Unknown ODE solver method: {method}")
        self.method = method

    def __call__(self, dydx, y0):
        if self.method == 'scipy_ivp':
            soln = scipy.integrate.solve_ivp(dydx, (self.xnow, self.xstop), y0)
            return soln.y[:, -1]
        else:
            soln = odeint(y0, self.xnow, self.xstop, dydx)
            return soln[0]

def DP(k, derivs, solver):
    y0 = np.exp(solver.xnow) * np.ones(2)
    return solver(lambda x, y: derivs.firstOrder(x, y, k), y0)

def growth_factor(k, derivs, solver):
    y = DP(k, derivs, solver)
    return y[1] / y[0]

def D2v2(kf, k1, k2, derivs, solver):
    y0 = np.empty(8)
    y0[:4] = np.exp(solver.xnow)
    y0[4:] = 3 * np.exp(2 * solver.xnow) / 7
    y0[5::2] *= 2
    return solver(lambda x, y: derivs.secondOrder(x, y, kf, k1, k2), y0)

def D3v2(x, k, p, derivs, solver):
    y0 = np.empty(10)
    y0[:4] = np.exp(solver.xnow)
    y0[4:8] = 3.0 * np.exp(2.0 * solver.xnow) / 7.0 * (1.0 - np.square(x))
    y0[5:8:2] *= 2.0
    y0[8] = (5.0 / (7.0 * 9.0)) * np.exp(3.0 * solver.xnow) * np.square(1.0 - np.square(x)) * (
        1.0 / (1.0 + np.square(p / k) + 2.0 * (p / k) * x)
        + 1.0 / (1.0 + np.square(p / k) - 2.0 * (p / k) * x)
    )
    y0[9] = (15.0 / (7.0 * 9.0)) * np.exp(3.0 * solver.xnow) * np.square(1.0 - np.square(x)) * (
        1.0 / (1.0 + np.square(p / k) + 2.0 * (p / k) * x)
        + 1.0 / (1.0 + np.square(p / k) - 2.0 * (p / k) * x)
    )
    return solver(lambda eta, y: derivs.thirdOrder(eta, y, x, k, p), y0)

def kernel_constants(f0, derivs, solver):
    KMIN = 1e-20
    # 2nd order
    Dpk1D2, _, Dpk2D2, _, DA2D2, DA2primeD2, DB2D2, _ = D2v2(KMIN, KMIN, KMIN, derivs, solver)
    KA_LCDM = DA2D2 / ((3/7) * Dpk1D2 * Dpk2D2)
    KAp_LCDM = DA2primeD2 / ((3/7) * Dpk1D2 * Dpk2D2) - 2 * DA2D2 / ((3/7) * Dpk1D2 * Dpk2D2) * f0
    # 3rd order
    DpkD3, _, DppD3, _, D2fD3, _, D2mfD3, _, D3symmD3, D3symmprimeD3 = D3v2(1e-7, KMIN, KMIN, derivs, solver)
    KR1_LCDM = (21/5) * D3symmD3 / (DpkD3 * DppD3 * DppD3)
    KR1p_LCDM = (21/5) * D3symmprimeD3 / (DpkD3 * DppD3 * DppD3) / (3 * f0)
    return KA_LCDM, KAp_LCDM, KR1_LCDM, KR1p_LCDM

# The following implementation is a translation of the NR C code used in FKPT,
# provided to allow direct comparison for testing and validation purposes.

from math import copysign
from typing import Callable, Tuple, Optional, List

TINY = 1.0e-30

def odeint(
    ystart: np.ndarray,
    x1: float,
    x2: float,
    derivs: Callable[[float, np.ndarray], np.ndarray],
    eps: float = 1e-4,
    h1: float = 2./5.,
    hmin: float = 0.0,
    maxnsteps: int = 10000,
    *,
    dxsav: Optional[float] = None,
    kmax: int = 0,
) -> Tuple[np.ndarray, int, int, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Integrate y'(x) = f(x, y) from x1 to x2 with adaptive steps and quality control.

    Parameters
    ----------
    ystart : array_like (nvar,)
        Initial condition at x1. Will not be modified in-place.
    x1, x2 : float
        Integration limits.
    derivs : function
        derivs(x, y) -> dydx
    eps : float
        Desired relative accuracy (passed to stepper).
    h1 : float
        Initial step size guess (sign is adjusted to match x2 - x1).
    hmin : float
        Minimum allowed step size magnitude.
    maxnsteps : int
        Safety cap on number of steps.
    dxsav : float, optional
        If provided and kmax>0, save (x,y) every ~dxsav in |x|.
    kmax : int
        Max number of saved output points (0 disables saving).

    Returns
    -------
    yfinal : np.ndarray
        y at x = x2 (final state).
    nok : int
        Count of successful steps with hdid == htry.
    nbad : int
        Count of steps where htry was reduced (hdid != htry).
    xp : np.ndarray or None
        Saved x samples (None if kmax==0 or dxsav is None).
    yp : np.ndarray or None
        Saved y samples with shape (kount, nvar) aligned with xp.
    """
    y = np.array(ystart, dtype=float, copy=True)
    nvar = y.size
    x = float(x1)
    h = copysign(abs(h1), x2 - x1)

    nok = 0
    nbad = 0

    # Output sampling (mimics NR globals: xp, yp, dxsav, kmax, kount)
    save_output = (kmax > 0 and dxsav is not None and dxsav > 0.0)
    dxsav_val = dxsav if dxsav is not None else 1.0
    xsav = x - 2.0 * dxsav_val  # force save on first eligible step
    xp: List[float] = []
    yp: List[np.ndarray] = []
    kount = 0

    for _ in range(1, maxnsteps + 1):
        dydx = derivs(x, y)
        yscal = np.abs(y) + np.abs(dydx * h) + TINY

        # Save if enough distance since last save (and capacity remains)
        if save_output and (kount < kmax - 1) and (abs(x - xsav) > abs(dxsav_val)):
            xp.append(x)
            yp.append(y.copy())
            kount += 1
            xsav = x

        # Shorten last step to land exactly on x2
        if (x + h - x2) * (x + h - x1) > 0.0:
            h = x2 - x

        # Take a step with quality control
        ynew, xnew, hdid, hnext = rkqs(y, dydx, x, h, eps, yscal, derivs)

        # Bookkeeping: “good” vs “bad” steps
        # NR compares hdid == h literally; we allow machine epsilon slack.
        if abs(hdid - h) <= max(1.0, abs(h)) * 1e-15:
            nok += 1
        else:
            nbad += 1

        y, x = ynew, xnew

        # Reached (or passed) the end?
        if (x - x2) * (x2 - x1) >= 0.0:
            # Final save and return
            if save_output and kount < kmax:
                xp.append(x)
                yp.append(y.copy())
                kount += 1
            return y, nok, nbad, (np.array(xp) if save_output else None), (np.vstack(yp) if save_output else None)

        if abs(hnext) <= hmin:
            raise RuntimeError("Step size too small in odeint")

        h = hnext

    raise RuntimeError("Too many steps in routine odeint")

SAFETY = 0.9
PGROW  = -0.2
PSHRNK = -0.25
ERRCON = 1.89e-4

def rkck(
    y: np.ndarray,
    dydx: np.ndarray,
    x: float,
    h: float,
    derivs: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cash–Karp Runge–Kutta step.
    Returns:
      yout : 5th-order solution estimate
      yerr : error estimate (y5 - y4), used for step-size control
    """
    # Cash–Karp coefficients (match the C code literals)
    a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875

    b21 = 0.2

    b31, b32 = 3.0/40.0, 9.0/40.0

    b41, b42, b43 = 0.3, -0.9, 1.2

    b51, b52, b53, b54 = -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0

    b61 = 1631.0/55296.0
    b62 = 175.0/512.0
    b63 = 575.0/13824.0
    b64 = 44275.0/110592.0
    b65 = 253.0/4096.0

    # 5th-order solution (c*)
    c1, c3, c4, c6 = 37.0/378.0, 250.0/621.0, 125.0/594.0, 512.0/1771.0

    # Differences y5 - y4 (dc*)
    dc1 = c1 - 2825.0/27648.0
    dc3 = c3 - 18575.0/48384.0
    dc4 = c4 - 13525.0/55296.0
    dc5 = -277.0/14336.0
    dc6 = c6 - 0.25

    k1 = dydx
    k2 = derivs(x + a2*h, y + h*(b21*k1))
    k3 = derivs(x + a3*h, y + h*(b31*k1 + b32*k2))
    k4 = derivs(x + a4*h, y + h*(b41*k1 + b42*k2 + b43*k3))
    k5 = derivs(x + a5*h, y + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4))
    k6 = derivs(x + a6*h, y + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5))

    yout = y + h*(c1*k1 + c3*k3 + c4*k4 + c6*k6)
    yerr = h*(dc1*k1 + dc3*k3 + dc4*k4 + dc5*k5 + dc6*k6)

    return yout, yerr


def rkqs(
    y: np.ndarray,
    dydx: np.ndarray,
    x: float,
    htry: float,
    eps: float,
    yscal: np.ndarray,
    derivs: Callable[[float, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, float, float, float]:
    """
    Quality-controlled single step (Numerical Recipes 'rkqs') using 'rkck'.

    Parameters
    ----------
    y      : current state at x (not modified in-place)
    dydx   : derivative at (x, y)
    x      : current independent variable
    htry   : trial step size
    eps    : desired accuracy
    yscal  : scaling vector (typically |y| + |dydx*h| + TINY)
    derivs : function f(x, y) -> dy/dx

    Returns
    -------
    ynew  : state after accepted step
    xnew  : x + hdid
    hdid  : actual step taken
    hnext : proposed next step
    """
    h = float(htry)

    while True:
        ytemp, yerr = rkck(y, dydx, x, h, derivs)

        # errmax = max_i |yerr_i / yscal_i| / eps
        errmax = float(np.max(np.abs(yerr / yscal)))
        errmax /= eps

        if errmax <= 1.0:
            # accepted
            break

        # rejected: shrink h
        htemp = SAFETY * h * (errmax ** PSHRNK)
        if h >= 0.0:
            h = max(htemp, 0.1 * h)
        else:
            h = min(htemp, 0.1 * h)

        xnew = x + h
        if xnew == x:
            raise RuntimeError("stepsize underflow in rkqs")

    # propose next step
    if errmax > ERRCON:
        hnext = SAFETY * h * (errmax ** PGROW)
    else:
        hnext = 5.0 * h

    hdid = h
    xnew = x + hdid
    ynew = ytemp

    return ynew, xnew, hdid, hnext
