import numpy as np

import scipy.integrate


class ModelDerivatives:

    def __init__(self, invH0, fR0, om, ol, beta2=1.0/6.0, nHS=1, screening=1):
        self.invH0 = invH0
        self.fR0 = fR0
        self.om = om
        self.ol = ol
        self.beta2 = beta2
        self.nHS = nHS
        self.screening = screening

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
