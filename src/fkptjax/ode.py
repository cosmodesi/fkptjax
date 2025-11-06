import numpy as np

import scipy.integrate


class ModelDerivatives:

    def __init__(self, invH0, fR0, om, ol, beta2=1.0/6.0, nHS=1, screening=1, omegaBD=0.0):
        self.invH0 = invH0
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

def DP(k, derivs, zout, xnow=-4):
    xstop = np.log(1.0/(1.0+zout))
    xspan = (xnow, xstop)
    y0 = np.exp(xnow) * np.ones(2)
    soln = scipy.integrate.solve_ivp(lambda x, y: derivs.firstOrder(x, y, k), xspan, y0)
    return soln.y[:, -1]

def growth_factor(k, derivs, zout, xnow=-4):
    y = DP(k, derivs, zout, xnow)
    return y[1] / y[0]

def D2v2(kf, k1, k2, derivs, zout, xnow=-4):
    xstop = np.log(1.0/(1.0+zout))
    xspan = (xnow, xstop)
    y0 = np.empty(8)
    y0[:4] = np.exp(xnow)
    y0[4:] = 3 * np.exp(2 * xnow) / 7
    y0[5::2] *= 2
    soln = scipy.integrate.solve_ivp(lambda x, y: derivs.secondOrder(x, y, kf, k1, k2), xspan, y0)
    return soln.y[:,-1]

def kernel_constants(derivs, zout, f0, xnow=-4):
    KMIN = 1e-20
    Dpk1D2, _, Dpk2D2, _, DA2D2, DA2primeD2, DB2D2, _ = D2v2(KMIN, KMIN, KMIN, derivs, zout, xnow)
    KA_LCDM = DA2D2 / ((3/7) * Dpk1D2 * Dpk2D2)
    KAp_LCDM = DA2primeD2 / ((3/7) * Dpk1D2 * Dpk2D2) - 2 * DA2D2 / ((3/7) * Dpk1D2 * Dpk2D2) * f0
    return KA_LCDM, KAp_LCDM
