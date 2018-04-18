#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import itertools

import numpy as np
import scipy.sparse as spsp
from scipy.interpolate import interp1d

__all__ = ['PointSource']

__docformat__ = "restructuredtext en"


class SourceBase(object):
    """Base class for representing a source emitter on a grid.

    Methods
    -------
    f(t, **kwargs)
        Evaluate w on grid numerically, must be implemented by sub class.

    Notes
    -----
    `intensity` could conceivably become a function of time, in the future.

    """

    def __init__(self, **kwargs):
        """Constructor for the SourceBase class.
        """

        self.shot = None

    def get_source_count(self):
        return 1

    source_count = property(get_source_count, None, None, None)

    def set_shot(self, shot):
        self.shot = shot

    def reset_time_series(self, ts):
        pass

    def f(self, t, **kwargs):
        raise NotImplementedError('Evaluation function \'f\' must be implemented by subclass.')

    def w(self, *argsw, **kwargs):
        raise NotImplementedError('Wavelet function must be implemented at the subclass level.')

    # For subclasses to implement.
    def serialize_dict(self, *args, **kwargs):
        raise NotImplementedError()

    def unserialize_dict(self, d):
        raise NotImplementedError()


class PointSource(SourceBase):
    """Subclass of PointRepresentationBase and SourceBase for representing a
    point source emitter on a grid.

    Attributes
    ----------
    domain : pysit.Domain
        Inherited from base class.
    position : tuple of float
        Inherited from base class.
    sampling_operator : scipy.sparse matrix
        Inherited from base class.
    adjoint_sampling_operator : scipy.sparse matrix
        Inherited from base class.
    intensity : float, optional
        Intensity of the source wavelet.
    w : function or function object
        Function of time that produces the source wavelet.

    Methods
    -------
    f(t, **kwargs)
        Evaluate w(t)*delta(x-x') numerically.

    """

    def __init__(self, loc, wavelet, mode='q', **kwargs):
        """Constructor for the PointSource class.

        Parameters
        ----------
        loc : tuple of float
            Coordinates of the point in the physical coordinates of the domain.
        wavelet : function or function object
            Function of time that produces the source wavelet.
        mode : string, optional
            Mode of the source.
        **kwargs : dict, optional
            May be used to specify `approximation` and `approximation_width` to
            base class.
        """

        # Populate parameters from the base class.
        SourceBase.__init__(self, **kwargs)

        self.loc = loc
        self.wavelet = wavelet
        self.mode = mode

    def f(self, t=0.0, nu=None, **kwargs):
        """Evaluate source emitter at time t on numerical grid.

        Parameters
        ----------
        t : float
            Time at which to evaluate the source wavelet.
        **kwargs : dict, optional
            May pass additional parameters to the source wavelet call.

        Returns
        -------
        The function w evaluated on a grid as an ndarray shaped like the domain.
        """
        if nu is None:
            if self._sample_interp_method == 'sparse':
                return (self.adjoint_sampling_operator * (self.intensity * self.w(t, **kwargs))).toarray().reshape(
                    self.mesh.shape())
            else:
                return (self.adjoint_sampling_operator * (self.intensity * self.w(t, **kwargs))).reshape(
                    self.mesh.shape())
        else:
            if self._sample_interp_method == 'sparse':
                return (self.adjoint_sampling_operator * (self.intensity * self.w(nu=nu, **kwargs))).toarray().reshape(
                    self.mesh.shape())
            else:
                return (self.adjoint_sampling_operator * (self.intensity * self.w(nu=nu, **kwargs))).reshape(
                    self.mesh.shape())
