# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:58:34 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""
# %%
from __future__ import division
from __future__ import print_function
import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_pml_coeff(int nx, int nz, int pml, int npml, DTYPE_t fc, DTYPE_t dx, DTYPE_t dz, DTYPE_t dt, DTYPE_t vp0):
    """
    % ! SEISMIC_CPML Version 1.1.1, November 2009.
    % !
    % ! Copyright Universite de Pau et des Pays de l'Adour, CNRS and INRIA, France.
    % ! Contributor: Dimitri Komatitsch, dimitri DOT komatitsch aT univ-pau DOT fr
    % !
    % ! This software is a computer program whose purpose is to solve
    % ! the two-dimensional isotropic elastic wave equation
    % ! using a finite-difference method with Convolutional Perfectly Matched
    % ! Layer (C-PML) conditions.
    % !
    % ! This software is governed by the CeCILL license under French law and
    % ! abiding by the rules of distribution of free software. You can use,
    % ! modify and/or redistribute the software under the terms of the CeCILL
    % ! license as circulated by CEA, CNRS and INRIA at the following URL
    % ! "http://www.cecill.info".
    % !
    % ! As a counterpart to the access to the source code and rights to copy,
    % ! modify and redistribute granted by the license, users are provided only
    % ! with a limited warranty and the software's author, the holder of the
    % ! economic rights, and the successive licensors have only limited
    % ! liability.
    % !
    % ! In this respect, the user's attention is drawn to the risks associated
    % ! with loading, using, modifying and/or developing or reproducing the
    % ! software by the user in light of its specific status of free software,
    % ! that may mean that it is complicated to manipulate, and that also
    % ! therefore means that it is reserved for developers and experienced
    % ! professionals having in-depth computer knowledge. Users are therefore
    % ! encouraged to load and test the software's suitability as regards their
    % ! requirements in conditions enabling the security of their systems and/or
    % ! data to be ensured and, more generally, to use and operate it in the
    % ! same conditions as regards security.
    % !
    % ! The full text of the license is available at the end of this program
    % ! and in file "LICENSE".
    %
    % ! The C-PML implementation is based in part on formulas given in Roden and Gedney (2000).
    % ! If you use this code for your own research, please cite some (or all) of these
    % ! articles:
    % !
    % ! @ARTICLE{MaKoEz08,
    % ! author = {Roland Martin and Dimitri Komatitsch and Abdela\^aziz Ezziani},
    % ! title = {An unsplit convolutional perfectly matched layer improved at grazing
    % ! incidence for seismic wave equation in poroelastic media},
    % ! journal = {Geophysics},
    % ! year = {2008},
    % ! volume = {73},
    % ! pages = {T51-T61},
    % ! number = {4},
    % ! doi = {10.1190/1.2939484}}
    % !
    % ! @ARTICLE{MaKo09,
    % ! author = {Roland Martin and Dimitri Komatitsch},
    % ! title = {An unsplit convolutional perfectly matched layer technique improved
    % ! at grazing incidence for the viscoelastic wave equation},
    % ! journal = {Geophysical Journal International},
    % ! year = {2009},
    % ! volume = {179},
    % ! pages = {333-344},
    % ! number = {1},
    % ! doi = {10.1111/j.1365-246X.2009.04278.x}}
    % !
    % ! @ARTICLE{MaKoGe08,
    % ! author = {Roland Martin and Dimitri Komatitsch and Stephen D. Gedney},
    % ! title = {A variational formulation of a stabilized unsplit convolutional perfectly
    % ! matched layer for the isotropic or anisotropic seismic wave equation},
    % ! journal = {Computer Modeling in Engineering and Sciences},
    % ! year = {2008},
    % ! volume = {37},
    % ! pages = {274-304},
    % ! number = {3}}
    % !
    % ! @ARTICLE{KoMa07,
    % ! author = {Dimitri Komatitsch and Roland Martin},
    % ! title = {An unsplit convolutional {P}erfectly {M}atched {L}ayer improved
    % !          at grazing incidence for the seismic wave equation},
    % ! journal = {Geophysics},
    % ! year = {2007},
    % ! volume = {72},
    % ! number = {5},
    % ! pages = {SM155-SM167},
    % ! doi = {10.1190/1.2757586}}
    % !
    % ! The original CPML technique for Maxwell's equations is described in:
    % !
    % ! @ARTICLE{RoGe00,
    % ! author = {J. A. Roden and S. D. Gedney},
    % ! title = {Convolution {PML} ({CPML}): {A}n Efficient {FDTD} Implementation
    % !          of the {CFS}-{PML} for Arbitrary Media},
    % ! journal = {Microwave and Optical Technology Letters},
    % ! year = {2000},
    % ! volume = {27},
    % ! number = {5},
    % ! pages = {334-339},
    % ! doi = {10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A}}
    """

    cdef DTYPE_t npower = 2.0
    cdef DTYPE_t k_max_PML = 1.0
    cdef DTYPE_t alpha_max_PML = 2.0*np.pi*(fc/2.0)

    cdef double[::1] d_x = np.zeros(nx, dtype=DTYPE)
    cdef double[::1] k_x = np.ones(nx, dtype=DTYPE)
    cdef double[::1] alpha_x = np.zeros(nx, dtype=DTYPE)
    cdef double[::1] a_x = np.zeros(nx, dtype=DTYPE)
    cdef double[::1] b_x = np.ones(nx, dtype=DTYPE)
    cdef double[::1] d_x_half = np.zeros(nx, dtype=DTYPE)
    cdef double[::1] k_x_half = np.ones(nx, dtype=DTYPE)
    cdef double[::1] alpha_x_half = np.zeros(nx, dtype=DTYPE)
    cdef double[::1] a_x_half = np.zeros(nx, dtype=DTYPE)
    cdef double[::1] b_x_half = np.ones(nx, dtype=DTYPE)

    cdef double[::1] d_z = np.zeros(nz, dtype=DTYPE)
    cdef double[::1] k_z = np.ones(nz, dtype=DTYPE)
    cdef double[::1] alpha_z = np.zeros(nz, dtype=DTYPE)
    cdef double[::1] a_z = np.zeros(nz, dtype=DTYPE)
    cdef double[::1] b_z = np.ones(nz, dtype=DTYPE)
    cdef double[::1] d_z_half = np.zeros(nz, dtype=DTYPE)
    cdef double[::1] k_z_half = np.ones(nz, dtype=DTYPE)
    cdef double[::1] alpha_z_half = np.zeros(nz, dtype=DTYPE)
    cdef double[::1] a_z_half = np.zeros(nz, dtype=DTYPE)
    cdef double[::1] b_z_half = np.ones(nz, dtype=DTYPE)

    # thickness of the PML layer in meters
    cdef DTYPE_t thickness_PML_x = npml * dx
    cdef DTYPE_t thickness_PML_z = npml * dz
    cdef DTYPE_t Rcoef = 0.001

    cdef DTYPE_t d0_x = - (npower + np.float64(1.0)) * vp0 * np.log(Rcoef, dtype=DTYPE) / (np.float64(2.0) * thickness_PML_x)
    cdef DTYPE_t d0_z = - (npower + np.float64(1.0)) * vp0 * np.log(Rcoef, dtype=DTYPE) / (np.float64(2.0) * thickness_PML_z)

    # damping in the X direction

    # origin of the PML layer (position of right edge minus thickness, in meters)
    cdef DTYPE_t xoriginleft = thickness_PML_x
    cdef DTYPE_t xoriginright = (nx - np.float64(1.0))*dx - thickness_PML_x

    cdef int i
    cdef DTYPE_t xval, abscissa_in_PML, abscissa_normalized

    cdef int nz2 = np.round(nz/2)

    for i in range(nx):

        # abscissa of current grid point along the damping profile
        xval = dx * i

        # left edge
        if i < npml:

            # define damping profile at the grid points
            abscissa_in_PML = xoriginleft - xval
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized**npower
                k_x[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_x[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

            # define damping profile at half the grid points
            abscissa_in_PML = xoriginleft - (xval + dx/2.0)
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized**npower
                k_x_half[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_x_half[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

        # right edge
        if i > (nx - npml - 1):

            # define damping profile at the grid points
            abscissa_in_PML = xval - xoriginright
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x[i] = d0_x * abscissa_normalized**npower
                k_x[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_x[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

            # define damping profile at half the grid points
            abscissa_in_PML = xval + dx/2.0 - xoriginright
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_x
                d_x_half[i] = d0_x * abscissa_normalized**npower
                k_x_half[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_x_half[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

        #check for negative values
        if alpha_x[i] < 0:
            alpha_x[i] = 0

        if alpha_x_half[i] < 0:
            alpha_x_half[i] = 0

        b_x[i] = np.exp(- (d_x[i] / k_x[i] + alpha_x[i]) * dt, dtype=DTYPE)
        b_x_half[i] = np.exp(- (d_x_half[i] / k_x_half[i] + alpha_x_half[i]) * dt, dtype=DTYPE)

        # this to avoid division by zero outside the PML
        if np.abs(d_x[i]) > 1e-6:
            a_x[i] = d_x[i] * (b_x[i] - 1.0) / (k_x[i] * (d_x[i] + k_x[i] * alpha_x[i]))

        if np.abs(d_x_half[i]) > 1e-6:
            a_x_half[i] = d_x_half[i] * (b_x_half[i] - 1.0) / (k_x_half[i] * (d_x_half[i] + k_x_half[i] * alpha_x_half[i]))

    # damping in the Y direction

    # origin of the PML layer (position of right edge minus thickness, in meters)
    cdef DTYPE_t yoriginbottom = thickness_PML_z
    cdef DTYPE_t yorigintop = (nz-1)*dz - thickness_PML_z

    cdef DTYPE_t yval

    for i in range(nz):

        # abscissa of current grid point along the damping profile
        yval = dz * i

        # bottom edge
        if i < npml:

            # define damping profile at the grid points
            abscissa_in_PML = yoriginbottom - yval
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                d_z[i] = d0_z * abscissa_normalized**npower
                k_z[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_z[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

            # define damping profile at half the grid points
            abscissa_in_PML = yoriginbottom - (yval + dz/2.0)
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                d_z_half[i] = d0_z * abscissa_normalized**npower
                k_z_half[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_z_half[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

        # top edge
        if i > (nz - npml - 1):

            # define damping profile at the grid points
            abscissa_in_PML = yval - yorigintop
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                d_z[i] = d0_z * abscissa_normalized**npower
                k_z[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_z[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

            # define damping profile at half the grid points
            abscissa_in_PML = yval + dz/2.0 - yorigintop
            if abscissa_in_PML >= 0:
                abscissa_normalized = abscissa_in_PML / thickness_PML_z
                d_z_half[i] = d0_z * abscissa_normalized**npower
                k_z_half[i] = 1.0 + (k_max_PML - 1.0) * abscissa_normalized**npower
                alpha_z_half[i] = alpha_max_PML * (1.0 - abscissa_normalized) + 0.1 * alpha_max_PML

        #check for negative values
        if alpha_z[i] < 0:
            alpha_z[i] = 0

        if alpha_z_half[i] < 0:
            alpha_z_half[i] = 0

        b_z[i] = np.exp(- (d_z[i] / k_z[i] + alpha_z[i]) * dt, dtype=DTYPE)
        b_z_half[i] = np.exp(- (d_z_half[i] / k_z_half[i] + alpha_z_half[i]) * dt, dtype=DTYPE)

        # this to avoid division by zero outside the PML
        if np.abs(d_z[i]) > 1e-6:
            a_z[i] = d_z[i] * (b_z[i] - 1.0) / (k_z[i] * (d_z[i] + k_z[i] * alpha_z[i]))

        if np.abs(d_z_half[i]) > 1e-6:
            a_z_half[i] = d_z_half[i] * (b_z_half[i] - 1.0) / (k_z_half[i] * (d_z_half[i] + k_z_half[i] * alpha_z_half[i]))

    # generate nx*nz matrix of PML coefficients
    cdef double[:,::1] A_x = np.zeros((nx,nz), dtype=DTYPE)
    cdef double[:,::1] B_x = np.zeros_like(A_x)
    cdef double[:,::1] K_x = np.zeros_like(A_x)
    cdef double[:,::1] A_x_half = np.zeros_like(A_x)
    cdef double[:,::1] B_x_half = np.zeros_like(A_x)
    cdef double[:,::1] K_x_half = np.zeros_like(A_x)

    cdef double[:,::1] A_z = np.zeros_like(A_x)
    cdef double[:,::1] B_z = np.zeros_like(A_x)
    cdef double[:,::1] K_z = np.zeros_like(A_x)
    cdef double[:,::1] A_z_half = np.zeros_like(A_x)
    cdef double[:,::1] B_z_half = np.zeros_like(A_x)
    cdef double[:,::1] K_z_half = np.zeros_like(A_x)

    for i in range(nz):
        A_x[0:nx,i] = a_x
        B_x[:,i] = b_x
        K_x[:,i] = k_x

        A_x_half[:,i] = a_x_half
        B_x_half[:,i] = b_x_half
        K_x_half[:,i] = k_x_half

    for i in range(nx):
        A_z[i,:] = a_z
        B_z[i,:] = b_z
        K_z[i,:] = k_z

        A_z_half[i,:] = a_z_half
        B_z_half[i,:] = b_z_half
        K_z_half[i,:] = k_z_half

        if pml == 3:
            # Take out PML along top edge...
            # JR/MV
            for j in range(nz2):
                 A_z[i,j] = A_z[i,nz2]
                 B_z[i,j] = B_z[i,nz2]
                 K_z[i,j] = K_z[i,nz2]

                 A_z_half[i,j] = A_z_half[i,nz2]
                 B_z_half[i,j] = B_z_half[i,nz2]
                 K_z_half[i,j] = K_z_half[i,nz2]

    return A_x, B_x, K_x, A_x_half, B_x_half, K_x_half, A_z, B_z, K_z, A_z_half, B_z_half, K_z_half