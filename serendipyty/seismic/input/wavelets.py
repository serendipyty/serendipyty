#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import math

import numpy as np

DTYPE = np.float64

__all__ = ['WaveletBase', 'RickerWavelet']

_sqrt2 = math.sqrt(2.0)

class WaveletBase(object):
    """ Base class for source wavelets or profile functions.

    This is implemented as a function object, so the magic happens in the
    `__call__` member function.

    Methods
    -------

    __call__(self, t=None, nu=None, **kwargs)

    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError('')

    def __call__(self, t=None, **kwargs):
        """Callable object method for the seismic sources.

        Parameters
        ----------
        t : float, array-like
            Time(s) at which to evaluate wavelet.

        """

        if t is not None:
            return self._evaluate_time(t)
        else:
            raise ValueError('Either a time or frequency must be provided.')


class RickerWavelet(WaveletBase):
    """ Ricker wavelet.

    The Ricker wavelet is the negative 2nd derivative of a Gaussian [1]_.

    :param fc: central frequency

    References
    ----------

    [1] N.  Ricker, "The form and laws of propagation of seismic wavelets,"
    Geophysics, vol. 18, pp. 10-40, 1953.

    """

    # Not allowed to change the order for the RickerWavelet.
    @property
    def order(self):
        return 2

    @order.setter
    def order(self, n):
        pass

    def __init__(self, t, fc=20.0, delay=1.0, **kwargs):
        self.t = t
        self.fc = fc
        self.delay = delay

        t_source = 1/self.fc
        t0 = t_source*self.delay
        #t = np.linspace(0,(self.nt-1)*self.dt,self.nt, dtype=DTYPE)
        tau = np.pi*(t-t0)/t0
        a = 2.0
        self.wavelet = (1-a*tau*tau)*np.exp(-(a/2)*tau*tau)

    def _evaluate_time(self, it):
        return self.wavelet[it]
