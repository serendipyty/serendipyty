#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import math
import numpy as np

DTYPE = np.float64

__all__ = ['BaseWavelet', 'RickerWavelet', 'RickerIntegratedWavelet']

_sqrt2 = math.sqrt(2.0)


class BaseWavelet(object):
    r""" Base class for source wavelets.

    This is implemented as a function object, so the magic happens in the
    `__call__` member function.

    Methods
    -------
    __call__(self, t=None, nu=None, **kwargs)

    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('')

    def __call__(self, it=None, **kwargs):
        r"""Callable object method for the seismic sources.

        Parameters
        ----------
        it : int, array-like
            Index(es) at which to evaluate wavelet.

        """

        if it is not None:
            return self._evaluate_time(it)
        else:
            raise ValueError('A time must be provided.')


class RickerWavelet(BaseWavelet):
    r""" Ricker wavelet.

    The Ricker wavelet is the negative 2nd derivative of a Gaussian [1]_.

    Parameters
    ----------
    t : float, ndarray
        Time array.
    fc : float, optional
        Central (peak) frequency of the wavelet
    delay : float, optional
        Time delay to be applied to the wavelet. The default delay is 1/`fc`.

    Attributes
    ----------
    wavelet : float, ndarray
        Array that contains the wavelet.

    References
    ----------

    .. [1] N. Ricker, The form and laws of propagation of seismic wavelets,
       Geophysics, vol. 18, pp. 10-40, 1953.

    """

    def __init__(self, t, dt=None, fc=20.0, delay=None, **kwargs):
        self.t = t
        if dt is None:
            self.dt = np.diff(self.t)[0]
        else:
            self.dt = dt
        self.fc = fc
        self.nt = t.size
        if delay is None:
            self.delay = 1 / self.fc
        else:
            self.delay = delay

        t_source = self.t - self.delay
        tau = (np.pi * self.fc * t_source) ** 2
        a = 2.0
        self.wavelet = (1 - a * tau) * np.exp(-(a / 2) * tau)

    def _evaluate_time(self, it):
        return self.wavelet[it]


class RickerIntegratedWavelet(BaseWavelet):
    r""" Integral of the Ricker wavelet.

    The Ricker wavelet is the negative 2nd derivative of a Gaussian [1]_.

    Parameters
    ----------
    t : float, ndarray
        Time array.
    fc : float, optional
        Central (peak) frequency of the wavelet
    normalized : bool, optional
        Normalize maximum amplitude to 1
    delay : float, optional
        Time delay to be applied to the wavelet. The default delay is 1/`fc`.

    Attributes
    ----------
    wavelet : float, ndarray
        Array that contains the wavelet.

    References
    ----------

    .. [1] N. Ricker, The form and laws of propagation of seismic wavelets,
       Geophysics, vol. 18, pp. 10-40, 1953.

    """

    def __init__(self, t, dt=None, fc=20.0, delay=None, normalized=False, **kwargs):
        self.t = t
        self.fc = fc
        if dt is None:
            self.dt = np.diff(self.t)[0]
        else:
            self.dt = dt
        if normalized:
            self.coeff = np.sqrt(2.0) * np.pi * self.fc / np.exp(-0.5)
        else:
            self.coeff = 1.0
        self.nt = t.size
        if delay is None:
            self.delay = 1 / self.fc
        else:
            self.delay = delay

        t_source = self.t - self.delay
        tau = (np.pi * self.fc * t_source) ** 2
        a = 2.0
        self.wavelet = (self.coeff * t_source) * np.exp(-(a / 2) * tau)

    def _evaluate_time(self, it):
        return self.wavelet[it]