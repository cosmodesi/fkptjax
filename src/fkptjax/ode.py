import numpy as np

import scipy.integrate


def get_growth_factor(k, invH0, fR0, om, ol, zout, xnow=-4, beta2=1.0/6.0, nHS=1):

    def mass(eta):
        return (
            1 / invH0 * np.sqrt(1 / (2 * np.abs(fR0)))
            * np.pow(om * np.exp(-3 * eta) + 4 * ol, (2 + nHS) / 2)
            / np.pow(om + 4 * ol, (1 + nHS) / 2)
        )
    def mu(eta, k):
        k2 = np.square(k)
        return 1 + 2 * beta2 * k2 / (k2 + np.exp(2 * eta) * np.square(mass(eta)))
    def f1(eta):
        return 3.0 / (2.0 * (1.0 + ol / om * np.exp(3 * eta)))
    def derivsFirstOrder(x, y):
        f1x = f1(x)
        return np.array([y[1], f1x * mu(x, k) * y[0] - (2 - f1x) * y[1]])
    xstop = np.log(1.0/(1.0+zout))
    xspan = (xnow, xstop)
    y0 = np.exp(xnow) * np.ones(2)
    soln = scipy.integrate.solve_ivp(derivsFirstOrder, xspan, y0)
    return soln.y[1, -1] / soln.y[0, -1]
