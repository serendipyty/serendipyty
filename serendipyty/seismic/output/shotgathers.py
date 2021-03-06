#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import math
import numpy as np

DTYPE = np.float64

__all__ = ['BaseModel', 'AcousticModel']

_sqrt2 = math.sqrt(2.0)


class BaseModel(object):
    r""" Base class for venlocity and density models.

    This is implemented as a function object, so the magic happens in the
    `__call__` member function.

    Attributes
    ----------
    n : list of int
        Dimensions in (x, y, z)

    Methods
    -------
    __call__(self, t=None, nu=None, **kwargs)

    """

    def __init__(self, dx, dy, dz):

        self.n = (-1, -1, -1)

        # Discretization
        self.dx = dx
        if dy is None:
            self.dy = self.dx
        else:
            self.dy = dy
        if dz is None:
            self.dz = self.dx
        else:
            self.dz = dz
        #raise NotImplementedError('')

    # def __call__(self, it=None, **kwargs):
    #     r"""Callable object method for the seismic sources.
    #
    #     Parameters
    #     ----------
    #     it : int, array-like
    #         Index(es) at which to evaluate wavelet.
    #
    #     """
    #
    #     if it is not None:
    #         return self._evaluate_time(it)
    #     else:
    #         raise ValueError('A time must be provided.')


class AcousticModel(BaseModel):
    r""" Acoustic model.

    Velocity and density models for an acoustic medium.

    Parameters
    ----------
    dx : float
        Spatial discretization in the x direction.
    dz : float, optional
        Spatial discretization in the y direction.
    dz : float, optional
        Spatial discretization in the z direction.
    vp : float, ndarray
        Velocity model
    rho : float, ndarray
        Density  model

    """

    def __init__(self, dx, vp, rho, dy=None, dz=None):

        super().__init__(dx, dy, dz)

        self.type = 'Acoustic'

        # Model parameters
        self.vp = vp
        self.rho = rho

        # 2D or 3D
        if self.vp.ndim == 3:
            self.is3d = True
            # Model dimensions
            self.n = self.vp.shape
        else:
            self.is3d = False
            self.n[0], self.n[2] = self.vp.shape
            self.n[1] = 1

