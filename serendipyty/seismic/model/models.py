#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import math
import numpy as np

import serendipyty.seismic.vis.vis as vis

DTYPE = np.float64

__all__ = ['BaseModel', 'AcousticModel']

_sqrt2 = math.sqrt(2.0)


class BaseModel(object):
    r""" Base class for velocity and density models.

    This is implemented as a function object, so the magic happens in the
    `__call__` member function.

    Attributes
    ----------
    n : list of int
        Dimensions in (x, y, z)
    ndim : int
        Number of dimensions

    Methods
    -------
    __call__(self, t=None, nu=None, **kwargs)

    """

    def __init__(self, modeltype, dx, dy, dz):

        self.type = modeltype

        self.n = [-1, -1, -1]

        self.ndim = 0

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

    def plot(self, style=None, **kwargs):
        r""" Plot the model parameters.
        """
        vis.plot(self.model, style=self.type, **kwargs)


class AcousticModel(BaseModel):
    r""" Acoustic model.

    Velocity and density models for an acoustic medium.

    Parameters
    ----------
    dx : float
        Spatial discretization in the x direction.
    dy : float, optional
        Spatial discretization in the y direction.
    dz : float, optional
        Spatial discretization in the z direction.
    vp : float, ndarray
        Velocity model
    rho : float, ndarray
        Density model

    """

    def __init__(self, dx, vp, rho, dy=None, dz=None):

        super().__init__('Acoustic', dx, dy, dz)

        # 2D or 3D
        if vp.ndim == 3:
            self.is3d = True
            self.ndim = 3
            self.n = vp.shape
        else:
            self.is3d = False
            self.ndim = 2
            self.n[0], self.n[2] = vp.shape
            self.n[1] = 1

        # Model parameters
        self.model = np.zeros((*vp.shape, 2))
        self.model[..., 0] = vp
        self.model[..., 1] = rho

        # Vis parameters
