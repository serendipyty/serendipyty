#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['BaseSource', 'PointSource']

__docformat__ = "restructuredtext en"


class BaseSource(object):
    r"""Base class for representing a source emitter on a grid.

    Methods
    -------
    f(t, **kwargs)
        Evaluate w on grid numerically, must be implemented by sub class.

    Notes
    -----
    `intensity` could conceivably become a function of time, in the future.

    """

    def __init__(self, **kwargs):
        """Constructor for the BaseSource class.
        """

        self.shot = None
        self.wavelet = None

    def set_shot(self, shot):
        self.shot = shot

    def plot(self, tmax=None,
             aspect='auto', style='wavelet', figsize=None):
        r""" Create a plot

        """

        # Remove the ugly ticks
        plt.tick_params(
            which='both',   # both major and minor ticks are affected
            bottom=False,   # ticks along the bottom edge are off
            top=False,      # ticks along the top edge are off
            left=False,     # ticks along the left edge are off
            right=False     # ticks along the right edge are off
        )

        # Create plot
        line = plt.plot(self.wavelet.t, self.wavelet.wavelet, linewidth=3)
        plt.title('Source wavelet')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        if tmax is not None:
            plt.xlim((0, tmax))

        return line

    def reset_time_series(self, ts):
        pass

    def f(self, t, **kwargs):
        raise NotImplementedError('Evaluation function \'f\' must be implemented by subclass.')

    def w(self, *argsw, **kwargs):
        raise NotImplementedError('Wavelet function must be implemented at the subclass level.')


class PointSource(BaseSource):
    r"""Subclass of BaseSource for representing a
    point source emitter on a grid.

    Methods
    -------
    f(t, **kwargs)
        Evaluate w(t)*delta(x-x') numerically.

    """

    def __init__(self, loc, wavelet, mode='q'):
        r"""Constructor for the PointSource class.

        Parameters
        ----------
        loc : float, ndarray
            Location of the source in the physical coordinates of the domain.
            loc should be a (n x 3) ndarray, where n denotes
            the number of sources.
        wavelet : BaseWavelet
            Function of time that produces the source wavelet.
            If only one source location is present, then wavelet should be a (1 x nt) ndarray.
            If multiple (n) source locations are present, then wavelet can be a (1 x nt) ndarray
            or a (n x nt) ndarray. In the first case, the same wavelet will be used
            for all source locations.
        mode : string, optional
            Mode of the source. Monopole source: 'q', 0.
            Dipole sources: 'fx', 1, 'fy', 2, 'fz', 3.
        **kwargs : dict, optional
            May be used to specify `approximation` and `approximation_width` to
            base class.
        """

        self.loc = loc
        self.wavelet = wavelet
        self.nt = self.wavelet.nt

        # Check dimensions of the locations array
        if np.squeeze(self.loc).ndim == 1:
            self.ns = 1
        else:
            self.ns = self.loc.shape[0]
            if np.squeeze(wavelet.wavelet).ndim == 1:
                self.wavelet_dim = self.ns
            elif np.squeeze(wavelet.wavelet).ndim == self.ns:
                self.wavelet_dim = self.ns
            else:
                raise ValueError('Something is wrong with the dimensions of the wavelet!')

        # Add a check on mode
        self.mode = mode
