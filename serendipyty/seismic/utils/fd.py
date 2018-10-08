# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7 11:35:44 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from timeit import default_timer as timer

DTYPE = np.float64

def stability(vmax, dx, dt):
    """Compute maximum dt for a stable simulation.

    Parameters
    ----------
    vmax: float
        Maximum velocity of the medium
    dx: float
        Grid discretization.
    dt: float
        Temporal discretization.

    Returns
    -------
    dt_stable: float
        Maximum temporal discretization.
    """

    dz = dx
    operator_order_coeff = 1.0
    courant_num_dt = vmax * np.sqrt(1.0 / (dx**2) + 1.0 / (dz**2))

    dt_stable = operator_order_coeff / courant_num_dt
    if dt > dt_stable:
        print('The simulation will get unstable!')

    return dt_stable

def dispersion(vmin, dx, fc, coeff=2.0):
    """Compute maximum dt for a stable simulation.

    Parameters
    ----------
    vmin: float
        Minimum velocity of the medium
    dx: float
        Grid discretization.
    fc: float
        Central (peak) frequency of the source wavelet.
    coeff: float
        Coefficient to compute the maximum frequency of the wavelet:
        fmax = coeff * fc.

    Returns
    -------
    dt_stable: float
        Maximum temporal discretization.
    """

    fmax = coeff * fc
    dx_no_dispersion = vmin / fmax / 6.0
    if dx > dx_no_dispersion:
        print('The simulation will show dispersion!')

    return dx_no_dispersion


def fdtaylorcoeff(k, xbar, x):
    """Compute coefficients for finite difference approximation.

    Compute coefficients for finite difference approximation for the
    derivative of order k at xbar based on grid values at points in x.

    Notes
    -----
    This function returns a row vector c of dimension 1 by n, where n=length(x),
    containing coefficients to approximate :math:`u^{(k)}(xbar)`,
    the k'th derivative of u evaluated at xbar,  based on n values
    of u at x(1), x(2), ... x(n).
    If U is a column vector containing u(x) at these n points, then
    c*U will give the approximation to :math:`u^{(k)}(xbar)`.
    Note for k=0 this can be used to evaluate the interpolating polynomial
    itself.
    Requires length(x) > k.
    Usually the elements x(i) are monotonically increasing
    and x(1) <= xbar <= x(n), but neither condition is required.
    The x values need not be equally spaced but must be distinct.
    This program should give the same results as fdcoeffV.m, but for large
    values of n is much more stable numerically.
    Based on the program "weights" in [1]_.

    Note: Forberg's algorithm can be used to simultaneously compute the
    coefficients for derivatives of order 0, 1, ..., m where m <= n-1.
    This gives a coefficient matrix C(1:n,1:m) whose k'th column gives
    the coefficients for the k'th derivative.
    In this version we set m=k and only compute the coefficients for
    derivatives of order up to order k, and then return only the k'th column
    of the resulting C matrix (converted to a row vector).
    This routine is then compatible with fdcoeffV.
    It can be easily modified to return the whole array if desired.

    From  http://www.amath.washington.edu/~rjl/fdmbook/ (2007).

    References
    ----------
    .. [1] B. Fornberg, "Calculation of weights in finite difference formulas",
      SIAM Review 40 (1998), pp. 685-691.
    """

    n = x.shape[0]
    n2 = int(n/2)
    if k >= n:
       print('*** length(x) must be larger than k')

    # change to m=n-1 if you want to compute coefficients for all
    #possible derivatives.  Then modify to output all of C.
    m = k
    c1 = 1.0
    c4 = x[0] - xbar
    C = np.zeros((n, m+1))
    C[0, 0] = 1.0
    for i in range(1, n):
        mn = np.min((i, m))
#        print(mn)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - xbar
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2*c3
            if j == i-1:
                for k in range(mn, 0, -1):
                    C[i, k] = c1*(k*C[i-1, k-1] - c5*C[i-1, k])/c2
                C[i, 0] = -c1*c5*C[i-1, 0]/c2
            for k in range(mn, 0, -1):
                C[j, k] = (c4*C[j, k] - k*C[j, k-1])/c3
            C[j, 0] = c4*C[j, 0]/c3
        c1 = c2
    # last column of c gives desired row vector
    c = np.zeros((n))
    c[...] = C[:, -1]

    alpha = c[n2:]*(2*np.arange(1, n2+1)-1)
    print(np.sum(alpha))

    h = np.sum(np.abs(c[n2:-1]))
    return c, h
