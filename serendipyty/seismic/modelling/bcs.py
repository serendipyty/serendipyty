#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

__all__ = ['BaseBc', 'PmlBc']


class BaseBc(object):
    r""" Base class for boundary conditions.

    This class defines the basic parameters for the boundary conditions
    used in the modelling routines.

    Attributes
    ----------
    omp_num_threads : int, optional
        Number of computational threads

    """

    def __init__(self, omp_num_threads=1):
        pass


class PmlBc(object):
    r""" PMLs boundary conditions.

    This class defines the basic parameters for the PMLs boundary conditions.

    Parameters
    ----------
    npml : int
        Number of grid points.
    freesurface : bool
        If True, applies a free surface on the top edge.

    Notes
    -----
    The perfectly matched layer (PML) absorbing boundary conditions
    are implemented following [1]_.

    .. [1] D. Komatitsch and J. Tromp,
    "A perfectly matched layer absorbing boundary condition
    for the second-order seismic wave equation",
    Geophys. J. Int. (2003) 154, 146–153
    """

    def __init__(self, npml=20, freesurface=False):

        self.type = 'Pml'
        self.npml = npml
        self.freesurface = freesurface
