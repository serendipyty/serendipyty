# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 11:51:43 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

# %% Full model run on 2D staggered grid
# O(2,2)

#from __future__ import division
#from __future__ import print_function
from timeit import default_timer as timer
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange, parallel

import serendipyty.seismic.modelling.generate_pml_coeff as generate_pml_coeff

import sys
#sys.path.append('/w04d2/bfilippo/pythonwork/cylinalg')
#sys.path.append('/w04d2/bfilippo/pythonwork/cylinalg/cylinalg')
#import scipy_blas
#cimport scipy_blas

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calc_ebc_pml(np.ndarray[DTYPE_t, ndim=2, mode='c'] c, np.ndarray[DTYPE_t, ndim=2, mode='c'] rho, DTYPE_t dx,
              size_t[::1] src_loc, double[::1] wav_src, DTYPE_t fc, DTYPE_t dt, sourcetypestring,
              size_t[:,::1] semt_origins, size_t[:,::1] semt_locs,
              size_t[:,::1] srec_locs,
              np.ndarray[DTYPE_t, ndim=2, mode='c'] gf,
              int snap, int gather_vb,
              int num_threads, int ntmax,
              double[:,::1] p_inj_v=None,
              ibctypestring=None):

    cdef size_t a, i, j, k, js, ks, f

    cdef double[:,::1] gf_v = gf

    cdef double[::1,:] aa, b, cc
    cdef int mm, nn, kk, lda, ldb, ldc
    cdef double alpha, beta

    cdef int nsub = semt_origins.shape[0]

    cdef size_t cn0, cn1, cn2, cn3

    # Needed for staggering of output wave fields
    cdef size_t[:,::1] face_v = np.array([[1, 0, 0],
                                            [1, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 1],
                                            [0, 1, 0],
                                            [0, 1, 0],
                                            [1, 0, 0],
                                            [1, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 1]], dtype=np.uint)

    # Initialization
    cdef int sourcetype
    if type(sourcetypestring) is str:
        if sourcetypestring == 'q':
            sourcetype = 0
        elif sourcetypestring == 'fx':
            sourcetype = 1
        elif sourcetypestring == 'fz':
            sourcetype = 3
    else:
        if sourcetypestring in {0, 1, 6, 7}:
            sourcetype = 1 # fx
        elif sourcetypestring in {2, 3, 8, 9}:
            sourcetype = 3 # fz

    cdef int ibctype
    if type(ibctypestring) is str:
        if ibctypestring == 'free':
            ibctype = 0
        elif ibctypestring == 'rigid':
            ibctype = 1

    cdef size_t[::1] src_loc_v = np.array(src_loc)
    cdef size_t[:,::1] semt_locs_v = np.array(semt_locs)
    cdef size_t[:,::1] srec_locs_v = np.array(srec_locs)

    cdef int nx = c.shape[0]
    cdef int nz = c.shape[1]
    cdef int nt = wav_src.shape[0]
    cdef int nrec = srec_locs.shape[0]
    cdef int nemt = semt_locs.shape[0]

    cdef DTYPE_t dz = dx
    cdef DTYPE_t d2 = 1.0/(dx*dz)
    cdef DTYPE_t vp0 = c[1,1]

    # Initialize output wave fields
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] snap_p = np.zeros((nt, nx, nz), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] snap_vx = np.zeros_like(snap_p)
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] snap_vz = np.zeros_like(snap_p)

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rec_p = np.zeros((nt, nrec), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rec_vx = np.zeros_like(rec_p)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rec_vz
    if not(gather_vb):
        rec_vz = np.zeros_like(rec_p)

    cdef int nx_pml, nz_pml, xpad, zpad, pml, npml

    pml = 3
    npml = 30

    if pml == 3:
        # Free surface on top
        nx_pml, nz_pml = nx + 2*npml, nz + 1*npml
        xpad = npml
        zpad = 0
        src_loc_v[0] = src_loc_v[0] + npml
        for i in range(nemt):
            semt_locs_v[i,0] = semt_locs_v[i,0] + npml
            srec_locs_v[i,0] = srec_locs_v[i,0] + npml

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] c_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rho_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

    if pml == 3:
        c_pml = np.pad(c, ((npml, npml), (0, npml)), 'edge')
        rho_pml = np.pad(rho, ((npml, npml), (0, npml)), 'edge')

    print(nx, nz, nx_pml, nz_pml, xpad, zpad, pml, npml)

    # Initialize all the typed memoryviews
    cdef double[:,::1] c1_v = dt*rho_pml*(c_pml**2)
    cdef double[:,::1] c2_v = dt/rho_pml
    cdef double[:,::1] diffop_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] diffop_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

#    cdef double[:,::1] diffop_x_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
#    cdef double[:,::1] diffop_z_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
    cdef double[:,::1] diffop_p_x_v = np.zeros((nx_pml-1, nz_pml), dtype=DTYPE)
    cdef double[:,::1] diffop_p_z_v = np.zeros((nx_pml, nz_pml-1), dtype=DTYPE)
    cdef double[:,::1] p_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] px_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] pz_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] vx_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] vx_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] vz_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] vz_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] a_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] b_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] K_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] a_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] b_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] K_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] a_x_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] b_x_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] K_x_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] a_z_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] b_z_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef double[:,::1] K_z_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

    cdef double[:,:,::1] snap_p_v = snap_p
    cdef double[:,:,::1] snap_vx_v = snap_vx
    cdef double[:,:,::1] snap_vz_v = snap_vz

    cdef double[:,::1] rec_p_v = rec_p
    cdef double[:,::1] rec_vx_v = rec_vx
    cdef double[:,::1] rec_vz_v
    if not(gather_vb):
        rec_vz_v = rec_vz

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ibc_mv = np.zeros((nt*nemt), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] ibc2 = np.zeros((nt,nemt), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ibc3 = np.zeros((nemt), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ext_field = np.zeros((2*nrec), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] p_mv = np.zeros((nt,nemt*nt), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] p_mv2 = np.zeros((nt,nemt*nt), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] p_mv3 = np.zeros((nt,nt,nemt), dtype=DTYPE)
#    cdef double[::1] ibc_v = np.zeros((nt*nemt), dtype=DTYPE)
#    cdef double[:,::1] ibc_v = np.zeros((nt,nemt), dtype=DTYPE)
    cdef double[:,::1] ibc2_v = ibc2
    cdef double[::1] ibc3_v = ibc3
#    cdef double[:,::1] p_mv_v = p_mv
    cdef double[::1] ibc_mv_v = ibc_mv
    cdef double[::1] ext_field_v = ext_field
    cdef double[:,::1] ext_field_out_v = np.zeros((nt, 2*nrec), dtype=DTYPE)
#    cdef double[::1] ext_field_p_v = np.zeros((nrec), dtype=DTYPE)
#    cdef double[::1] ext_field_v_v = np.zeros((nrec), dtype=DTYPE)

#    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] aaa

    # Srec MASK
    # to be multiplied with extrapolation integral from Srec to Semt
    maskdict = {0: (-1, (1, 0)),
                1: (+1, (1, 0)),
                2: (-1, (0, 1)),
                3: (+1, (0, 1)),
                6: (-0.5, (1, 0)),
                7: (+0.5, (1, 0)),
                8: (-0.5, (0, 1)),
                9: (+0.5, (0, 1)),
                }

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] srec_mask = np.zeros((2*nrec), dtype=DTYPE)
    cdef double[::1] srec_mask_v = srec_mask

    for i, loc in enumerate(srec_locs):
        srec_mask_v[i] = maskdict[loc[3]][0]
        srec_mask_v[i+nrec] = maskdict[loc[3]][0]

    print('###################################################')
    print('###################################################')
    print('v13')
    print('###################################################')
    print('###################################################')


    # Compute PML coefficients
    a_x_v, b_x_v, K_x_v, a_x_half_v, b_x_half_v, K_x_half_v, a_z_v, b_z_v, K_z_v, a_z_half_v, b_z_half_v, K_z_half_v = \
        generate_pml_coeff.generate_pml_coeff(nx_pml, nz_pml, pml, npml, fc, dx, dz, dt, vp0)

    # Main loop
    start = timer()
#    for a in range(nt):
    for a in range(ntmax):

        if a%100 == 0:
            print('FULL model, timestepp = ', a, 1)

        # Inject velocity source
        if sourcetype == 1:
            vx_v[src_loc_v[0],src_loc_v[2]] = vx_v[src_loc_v[0],src_loc_v[2]] + c2_v[src_loc_v[0],src_loc_v[2]]*0.5*wav_src[a]*d2
            vx_v[src_loc_v[0]-1,src_loc_v[2]] = vx_v[src_loc_v[0]-1,src_loc_v[2]] + c2_v[src_loc_v[0]-1,src_loc_v[2]]*0.5*wav_src[a]*d2
        elif sourcetype == 3:
            vz_v[src_loc_v[0],src_loc_v[2]] = vz_v[src_loc_v[0],src_loc_v[2]] + c2_v[src_loc_v[0],src_loc_v[2]]*0.5*wav_src[a]*d2
            vz_v[src_loc_v[0],src_loc_v[2]-1] = vz_v[src_loc_v[0],src_loc_v[2]-1] + c2_v[src_loc_v[0],src_loc_v[2]-1]*0.5*wav_src[a]*d2

        # Inject pressure source
        if sourcetype == 0:
            p_v[src_loc_v[0],src_loc_v[2]] = p_v[src_loc_v[0],src_loc_v[2]] + c1_v[src_loc_v[0],src_loc_v[2]]*wav_src[a]*d2

        # Update vx and vz
        for n in range(nsub):
#            cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(cn0,cn1, schedule='dynamic'):
#                    for j in range(cn2,cn3):
#                        vx_v[i,j] = vx_v[i,j] + c2_v[i,j] * (p_v[i+1,j] - p_v[i,j]) / dx

            cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nx_pml-1, schedule='dynamic'):
                    for j in range(cn2,nz_pml):
                        diffop_p_x_v[i,j] = (p_v[i+1,j] - p_v[i,j]) / dx
                        px_mem_v[i,j] = b_x_half_v[i,j] * px_mem_v[i,j] + a_x_half_v[i,j] * diffop_p_x_v[i,j]
                        diffop_p_x_v[i,j] = diffop_p_x_v[i,j] / K_x_half_v[i,j] + px_mem_v[i,j]
                        vx_v[i,j] = vx_v[i,j] + c2_v[i,j] * diffop_p_x_v[i,j]

#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(cn0,cn1, schedule='dynamic'):
#                    for j in range(cn2,cn3):
#                        vz_v[i,j] = vz_v[i,j] + c2_v[i,j] * (p_v[i,j+1] - p_v[i,j]) / dz

            with nogil, parallel(num_threads=num_threads):
                for i in prange(nx_pml, schedule='dynamic'):
                    for j in range(cn2,nz_pml-1):
                        diffop_p_z_v[i,j] = (p_v[i,j+1] - p_v[i,j]) / dx
                        pz_mem_v[i,j] = b_z_half_v[i,j] * pz_mem_v[i,j] + a_z_half_v[i,j] * diffop_p_z_v[i,j]
                        diffop_p_z_v[i,j] = diffop_p_z_v[i,j] / K_z_half_v[i,j] + pz_mem_v[i,j]
                        vz_v[i,j] = vz_v[i,j] + c2_v[i,j] * diffop_p_z_v[i,j]

        # Rigid boundary
        if ibctype == 1:
            for n in range(nsub):
                cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
                # on 2 and 3
                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn0+xpad,cn1+xpad, schedule='dynamic'):
                    for i in prange(nx_pml, schedule='dynamic'):
                        vz_v[i,cn2-1] = -vz_v[i,cn2]
#                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn0,cn1, schedule='dynamic'):
#                        vz_v[i,cn3-1] = -vz_v[i,cn3-2]
                # on 0 and 1
#                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn2,cn3, schedule='dynamic'):
#                        vx_v[cn0-1,i] = -vx_v[cn0,i]
#                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn2,cn3, schedule='dynamic'):
#                        vx_v[cn1-1,i] = -vx_v[cn1-2,i]

        # INJECTION of MPS
        # Dipole sources
        if ibctype == 0 and a > 0:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nemt, schedule='dynamic'):
                    if semt_locs_v[i,3] in {0, 6}:
                        vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] - 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(p_inj_v[a-1,i]) / dx
                    elif semt_locs_v[i,3] in {1, 7}:
                        vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(p_inj_v[a-1,i]) / dx
                    elif semt_locs_v[i,3] in {2, 8}:
                        vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] - 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(p_inj_v[a-1,i]) / dx
                    elif semt_locs_v[i,3] in {3, 9}:
                        vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(p_inj_v[a-1,i]) / dx

       # INJECTION of IBCs
        # Dipole sources
        if ibctype == 0:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nemt, schedule='dynamic'):
                    if semt_locs_v[i,3] in {0, 6}:
                        vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] - 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(ibc3_v[i])
                    elif semt_locs_v[i,3] in {1, 7}:
                        vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vx_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(ibc3_v[i])
                    elif semt_locs_v[i,3] in {2, 8}:
                        vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] - 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(ibc3_v[i])
                    elif semt_locs_v[i,3] in {3, 9}:
                        vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] = vz_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c2_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(ibc3_v[i])

        # Update p
        for n in range(nsub):
            cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(cn0,cn1, schedule='dynamic'):
#                    for j in range(cn2,cn3):
#                        diffop_x_v[i,j] = (vx_v[i,j] - vx_v[i-1,j]) / dx
#                        diffop_z_v[i,j] = (vz_v[i,j] - vz_v[i,j-1]) / dz
#                        p_v[i,j] = p_v[i,j] + c1_v[i,j] * (diffop_x_v[i,j] + diffop_z_v[i,j])

            with nogil, parallel(num_threads=num_threads):
                for i in prange(1,nx_pml-1, schedule='dynamic'):
                #for i in range(nx_pml-1):
                    for j in range(cn2,nz_pml-1):
                        diffop_x_v[i,j] = (vx_v[i,j] - vx_v[i-1,j]) / dx
                        vx_mem_v[i,j+1] = b_x_v[i,j+1] * vx_mem_v[i,j+1] + a_x_v[i,j+1] * diffop_x_v[i,j]
                        diffop_x_v[i,j] = diffop_x_v[i,j] / K_x_v[i,j+1] + vx_mem_v[i,j+1]
                        diffop_z_v[i,j] = (vz_v[i,j] - vz_v[i,j-1]) / dx
                        vz_mem_v[i+1,j] = b_z_v[i+1,j] * vz_mem_v[i+1,j] + a_z_v[i+1,j] * diffop_z_v[i,j]
                        diffop_z_v[i,j] = diffop_z_v[i,j] / K_z_v[i+1,j] + vz_mem_v[i+1,j]
                        p_v[i,j] = p_v[i,j] + c1_v[i,j] * (diffop_x_v[i,j] + diffop_z_v[i,j])

        # Free surface
        if ibctype == 0:
            for n in range(nsub):
                cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
                # on 2 and 3
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn0,cn1, schedule='dynamic'):
                        p_v[i,cn2] = -p_v[i,cn2+1]
#                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn0,cn1, schedule='dynamic'):
#                        p_v[i,cn3] = -p_v[i,cn3-1]
                # on 0 and 1
#                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn2,cn3, schedule='dynamic'):
#                        p_v[cn0,i] = -p_v[cn0+1,i]
#                with nogil, parallel(num_threads=num_threads):
#                    for i in prange(cn2,cn3, schedule='dynamic'):
#                        p_v[cn1,i] = -p_v[cn1-1,i]

        # PREDICTION
        # Predict pressure on Semt
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nrec, schedule='dynamic'):
                ext_field_v[i] = p_v[srec_locs_v[i,0],srec_locs_v[i,2]] #*srec_mask_v[i]
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nrec, schedule='dynamic'):
                if srec_locs_v[i,3] in {0, 1, 6, 7}:
                    ext_field_v[i+nrec] = 0.5 * (vx_v[srec_locs_v[i,0]-1,srec_locs_v[i,2]] + vx_v[srec_locs_v[i,0],srec_locs_v[i,2]]) #*srec_mask_v[i]
                elif srec_locs_v[i,3] in {2, 3, 8, 9}:
                    ext_field_v[i+nrec] = 0.5 * (vz_v[srec_locs_v[i,0],srec_locs_v[i,2]-1] + vz_v[srec_locs_v[i,0],srec_locs_v[i,2]]) #*srec_mask_v[i]

        ext_field[:] = ext_field * srec_mask
#        alpha = 1.0
#        beta = 0.0
#        lda = nt*nemt
#        ldb = 1 #2*nrec
#        ldc = 1 #nt*nemt
#        mm = nt*nemt
#        kk = 2*nrec
#        scipy_blas.dgemv("N",  &mm, &kk, &alpha, &gff_v[0,0], &lda, &ext_field_v[0], &ldb, &beta, &ibc_mv_v[0], &ldc)
        ibc_mv = np.dot(gf_v,ext_field)
#        ibc2[...] = ibc2 + np.roll(ibc_mv.reshape((nemt,nt)).T, a, axis=0) / dx
        ibc3[:] = ibc2[a,:]
        ibc2[...] = ibc2 - np.roll(ibc_mv.reshape((nt,nemt), order='F'), a+1, axis=0)

        # INJECTION of EBCs
        # Monopole sources
        if ibctype == 1:
#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(nemt, schedule='dynamic'):
                for i in range(nemt):
                    p_v[semt_locs_v[i,0],semt_locs_v[i,2]] = p_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c1_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(ibc3_v[i])

        # INJECTION of MPS
        # Monopole sources
        if ibctype == 1:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nemt, schedule='dynamic'):
                    if semt_locs_v[i,3] in {0, 2, 6, 8}:
                        p_v[semt_locs_v[i,0],semt_locs_v[i,2]] = p_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c1_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(p_inj_v[a,i]) / dx
                    elif semt_locs_v[i,3] in {1, 3, 7, 9}:
                        p_v[semt_locs_v[i,0],semt_locs_v[i,2]] = p_v[semt_locs_v[i,0],semt_locs_v[i,2]] + 2 * c1_v[semt_locs_v[i,0],semt_locs_v[i,2]]*(p_inj_v[a,i]) / dx

        # Save snapshots
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nx, schedule='dynamic'):
                for j in range(nz):
                    snap_p_v[a,i,j] = p_v[i+npml,j]
                    snap_vx_v[a,i,j] = vx_v[i+npml,j]
                    snap_vz_v[a,i,j] = vz_v[i+npml,j]
        # Save receiver wave fields
#        with nogil, parallel(num_threads=num_threads):
#            for i in prange(nrec, schedule='dynamic'):
#                f = rec_loc[i,3]
#                j = rec_loc[i,0]
#                js = face_v[f,0]
#                k = rec_loc[i,2]
#                ks = face_v[f,2]
#                rec_p_v[a,i] = 0.5 * (p_v[j,k] + p_v[j+js,k+ks])
#                if f == 0 or f == 1 or f == 6 or f == 7:
#                    rec_vx_v[a,i] = 0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
#                elif f == 2 or f == 3 or f == 8 or f == 9:
#                    rec_vx_v[a,i] = 0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])

    end = timer()
    print(end - start, 's')

#    return snap_p_v, snap_vx_v, snap_vz_v, ibc_v, ibc2, ext_field_out_v, p_mv, p_mv2, p_mv3
    return snap_p, snap_vx, snap_vz
#    if snap and not(gather_vb):
#        return rec_p, rec_vx, rec_vz, snap_p, snap_vx, snap_vz
#    elif not(snap) and not(gather_vb):
#        return rec_p, rec_vx, rec_vz
#    elif not(snap) and gather_vb:
#        return rec_p, rec_vx
#    else:
#        print('Something went wrong!')
#        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calc_ebc(np.ndarray[DTYPE_t, ndim=2, mode='c'] c, np.ndarray[DTYPE_t, ndim=2, mode='c'] rho, DTYPE_t dx,
              size_t[::1] src_loc, double[::1] wav_src, DTYPE_t fc, DTYPE_t dt, sourcetypestring,
              size_t[:,::1] semt_origins, size_t[:,::1] semt_locs,
              size_t[:,::1] srec_locs,
              np.ndarray[DTYPE_t, ndim=2, mode='c'] gf,
              int snap, int gather_vb,
              int num_threads,
              double[:,::1] p_inj_v=None,
              ibctypestring=None):

    cdef size_t a, i, j, k, js, ks, f

    cdef double[:,::1] gf_v = gf

    cdef double[::1,:] aa, b, cc
    cdef int mm, nn, kk, lda, ldb, ldc
    cdef double alpha, beta

    cdef int nsub = semt_origins.shape[0]

    cdef size_t cn0, cn1, cn2, cn3

    # Needed for staggering of output wave fields
    cdef size_t[:,::1] face_v = np.array([[1, 0, 0],
                                            [1, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 1],
                                            [0, 1, 0],
                                            [0, 1, 0],
                                            [1, 0, 0],
                                            [1, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 1]], dtype=np.uint)

    # Initialization
    cdef int sourcetype
    if type(sourcetypestring) is str:
        if sourcetypestring == 'q':
            sourcetype = 0
        elif sourcetypestring == 'fx':
            sourcetype = 1
        elif sourcetypestring == 'fz':
            sourcetype = 3
    else:
        if sourcetypestring in {0, 1, 6, 7}:
            sourcetype = 1 # fx
        elif sourcetypestring in {2, 3, 8, 9}:
            sourcetype = 3 # fz

    cdef int ibctype
    if type(ibctypestring) is str:
        if ibctypestring == 'free':
            ibctype = 0
        elif ibctypestring == 'rigid':
            ibctype = 1

    cdef size_t[::1] src_loc_v = np.array(src_loc)

    cdef int nx = c.shape[0]
    cdef int nz = c.shape[1]
    cdef int nt = wav_src.shape[0]
    cdef int nrec = srec_locs.shape[0]
    cdef int nemt = semt_locs.shape[0]

    cdef DTYPE_t dz = dx
    cdef DTYPE_t d2 = 1.0/(dx*dz)
    cdef DTYPE_t vp0 = c[1,1]

    # Initialize output wave fields
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] snap_p = np.zeros((nt, nx, nz), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] snap_vx = np.zeros_like(snap_p)
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] snap_vz = np.zeros_like(snap_p)

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rec_p = np.zeros((nt, nrec), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rec_vx = np.zeros_like(rec_p)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rec_vz
    if not(gather_vb):
        rec_vz = np.zeros_like(rec_p)

    # Initialize all the typed memoryviews
    cdef double[:,::1] c1_v = dt*rho*(c**2)
    cdef double[:,::1] c2_v = dt/rho
    cdef double[:,::1] p_v = np.zeros((nx, nz), dtype=DTYPE)
    cdef double[:,::1] vx_v = np.zeros((nx, nz), dtype=DTYPE)
    cdef double[:,::1] vz_v = np.zeros((nx, nz), dtype=DTYPE)
    cdef double[:,::1] diffop_x_v = np.zeros((nx, nz), dtype=DTYPE)
    cdef double[:,::1] diffop_z_v = np.zeros((nx, nz), dtype=DTYPE)

    cdef double[:,:,::1] snap_p_v = snap_p
    cdef double[:,:,::1] snap_vx_v = snap_vx
    cdef double[:,:,::1] snap_vz_v = snap_vz

    cdef double[:,::1] rec_p_v = rec_p
    cdef double[:,::1] rec_vx_v = rec_vx
    cdef double[:,::1] rec_vz_v
    if not(gather_vb):
        rec_vz_v = rec_vz

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ibc_mv = np.zeros((nt*nemt), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] ibc2 = np.zeros((nt,nemt), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ibc3 = np.zeros((nemt), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] ext_field = np.zeros((2*nrec), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] p_mv = np.zeros((nt,nemt*nt), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] p_mv2 = np.zeros((nt,nemt*nt), dtype=DTYPE)
#    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] p_mv3 = np.zeros((nt,nt,nemt), dtype=DTYPE)
#    cdef double[::1] ibc_v = np.zeros((nt*nemt), dtype=DTYPE)
#    cdef double[:,::1] ibc_v = np.zeros((nt,nemt), dtype=DTYPE)
    cdef double[:,::1] ibc2_v = ibc2
    cdef double[::1] ibc3_v = ibc3
#    cdef double[:,::1] p_mv_v = p_mv
    cdef double[::1] ibc_mv_v = ibc_mv
    cdef double[::1] ext_field_v = ext_field
    cdef double[:,::1] ext_field_out_v = np.zeros((nt, 2*nrec), dtype=DTYPE)
#    cdef double[::1] ext_field_p_v = np.zeros((nrec), dtype=DTYPE)
#    cdef double[::1] ext_field_v_v = np.zeros((nrec), dtype=DTYPE)

#    cdef np.ndarray[np.uint_t, ndim=1, mode='c'] aaa

    # Srec MASK
    # to be multiplied with extrapolation integral from Srec to Semt
    maskdict = {0: (-1, (1, 0)),
                1: (+1, (1, 0)),
                2: (-1, (0, 1)),
                3: (+1, (0, 1)),
                6: (-0.5, (1, 0)),
                7: (+0.5, (1, 0)),
                8: (-0.5, (0, 1)),
                9: (+0.5, (0, 1)),
                }

    cdef np.ndarray[DTYPE_t, ndim=1, mode='c'] srec_mask = np.zeros((2*nrec), dtype=DTYPE)
    cdef double[::1] srec_mask_v = srec_mask

    for i, loc in enumerate(srec_locs):
        srec_mask_v[i] = maskdict[loc[3]][0]
        srec_mask_v[i+nrec] = maskdict[loc[3]][0]

    print('###################################################')
    print('###################################################')
    print('v13')
    print('###################################################')
    print('###################################################')

    # Main loop
    start = timer()
#    for a in range(nt):
    for a in range(400):

        if a%100 == 0:
            print('FULL model, timestepp = ', a, 1)

        # Inject velocity source
        if sourcetype == 1:
            vx_v[src_loc_v[0],src_loc_v[2]] = vx_v[src_loc_v[0],src_loc_v[2]] + c2_v[src_loc_v[0],src_loc_v[2]]*0.5*wav_src[a]*d2
            vx_v[src_loc_v[0]-1,src_loc_v[2]] = vx_v[src_loc_v[0]-1,src_loc_v[2]] + c2_v[src_loc_v[0]-1,src_loc_v[2]]*0.5*wav_src[a]*d2
        elif sourcetype == 3:
            vz_v[src_loc_v[0],src_loc_v[2]] = vz_v[src_loc_v[0],src_loc_v[2]] + c2_v[src_loc_v[0],src_loc_v[2]]*0.5*wav_src[a]*d2
            vz_v[src_loc_v[0],src_loc_v[2]-1] = vz_v[src_loc_v[0],src_loc_v[2]-1] + c2_v[src_loc_v[0],src_loc_v[2]-1]*0.5*wav_src[a]*d2

        # Inject pressure source
        if sourcetype == 0:
            p_v[src_loc_v[0],src_loc_v[2]] = p_v[src_loc_v[0],src_loc_v[2]] + c1_v[src_loc_v[0],src_loc_v[2]]*wav_src[a]*d2

        # Update vx and vz
        for n in range(nsub):
            cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
            with nogil, parallel(num_threads=num_threads):
                for i in prange(cn0,cn1, schedule='dynamic'):
                    for j in range(cn2,cn3):
                        vx_v[i,j] = vx_v[i,j] + c2_v[i,j] * (p_v[i+1,j] - p_v[i,j]) / dx

            with nogil, parallel(num_threads=num_threads):
                for i in prange(cn0,cn1, schedule='dynamic'):
                    for j in range(cn2,cn3):
                        vz_v[i,j] = vz_v[i,j] + c2_v[i,j] * (p_v[i,j+1] - p_v[i,j]) / dz

        # Rigid boundary
        if ibctype == 1:
            for n in range(nsub):
                cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
                # on 2 and 3
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn0,cn1, schedule='dynamic'):
                        vz_v[i,cn2-1] = -vz_v[i,cn2]
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn0,cn1, schedule='dynamic'):
                        vz_v[i,cn3-1] = -vz_v[i,cn3-2]
                # on 0 and 1
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn2,cn3, schedule='dynamic'):
                        vx_v[cn0-1,i] = -vx_v[cn0,i]
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn2,cn3, schedule='dynamic'):
                        vx_v[cn1-1,i] = -vx_v[cn1-2,i]

        # INJECTION of MPS
        # Dipole sources
        if ibctype == 0 and a > 0:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nemt, schedule='dynamic'):
                    if semt_locs[i,3] in {0, 6}:
                        vx_v[semt_locs[i,0],semt_locs[i,2]] = vx_v[semt_locs[i,0],semt_locs[i,2]] - 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(p_inj_v[a-1,i]) / dx
                    elif semt_locs[i,3] in {1, 7}:
                        vx_v[semt_locs[i,0],semt_locs[i,2]] = vx_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(p_inj_v[a-1,i]) / dx
                    elif semt_locs[i,3] in {2, 8}:
                        vz_v[semt_locs[i,0],semt_locs[i,2]] = vz_v[semt_locs[i,0],semt_locs[i,2]] - 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(p_inj_v[a-1,i]) / dx
                    elif semt_locs[i,3] in {3, 9}:
                        vz_v[semt_locs[i,0],semt_locs[i,2]] = vz_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(p_inj_v[a-1,i]) / dx

       # INJECTION of IBCs
        # Dipole sources
        if ibctype == 0:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nemt, schedule='dynamic'):
                    if semt_locs[i,3] in {0, 6}:
                        vx_v[semt_locs[i,0],semt_locs[i,2]] = vx_v[semt_locs[i,0],semt_locs[i,2]] - 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(ibc3_v[i])
                    elif semt_locs[i,3] in {1, 7}:
                        vx_v[semt_locs[i,0],semt_locs[i,2]] = vx_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(ibc3_v[i])
                    elif semt_locs[i,3] in {2, 8}:
                        vz_v[semt_locs[i,0],semt_locs[i,2]] = vz_v[semt_locs[i,0],semt_locs[i,2]] - 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(ibc3_v[i])
                    elif semt_locs[i,3] in {3, 9}:
                        vz_v[semt_locs[i,0],semt_locs[i,2]] = vz_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c2_v[semt_locs[i,0],semt_locs[i,2]]*(ibc3_v[i])

        # Update p
        for n in range(nsub):
            cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
            with nogil, parallel(num_threads=num_threads):
                for i in prange(cn0,cn1, schedule='dynamic'):
                    for j in range(cn2,cn3):
                        diffop_x_v[i,j] = (vx_v[i,j] - vx_v[i-1,j]) / dx
                        diffop_z_v[i,j] = (vz_v[i,j] - vz_v[i,j-1]) / dz
                        p_v[i,j] = p_v[i,j] + c1_v[i,j] * (diffop_x_v[i,j] + diffop_z_v[i,j])

        # Free surface
        if ibctype == 0:
            for n in range(nsub):
                cn0, cn1, cn2, cn3 = semt_origins[n][0], semt_origins[n][0] + semt_origins[n][2], semt_origins[n][1], semt_origins[n][1] + semt_origins[n][3]
                # on 2 and 3
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn0,cn1, schedule='dynamic'):
                        p_v[i,cn2] = -p_v[i,cn2+1]
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn0,cn1, schedule='dynamic'):
                        p_v[i,cn3] = -p_v[i,cn3-1]
                # on 0 and 1
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn2,cn3, schedule='dynamic'):
                        p_v[cn0,i] = -p_v[cn0+1,i]
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(cn2,cn3, schedule='dynamic'):
                        p_v[cn1,i] = -p_v[cn1-1,i]

        # PREDICTION
        # Predict pressure on Semt
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nrec, schedule='dynamic'):
                ext_field_v[i] = p_v[srec_locs[i,0],srec_locs[i,2]] #*srec_mask_v[i]
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nrec, schedule='dynamic'):
                if srec_locs[i,3] in {0, 1, 6, 7}:
                    ext_field_v[i+nrec] = 0.5 * (vx_v[srec_locs[i,0]-1,srec_locs[i,2]] + vx_v[srec_locs[i,0],srec_locs[i,2]]) #*srec_mask_v[i]
                elif srec_locs[i,3] in {2, 3, 8, 9}:
                    ext_field_v[i+nrec] = 0.5 * (vz_v[srec_locs[i,0],srec_locs[i,2]-1] + vz_v[srec_locs[i,0],srec_locs[i,2]]) #*srec_mask_v[i]

        ext_field[:] = ext_field * srec_mask
#        alpha = 1.0
#        beta = 0.0
#        lda = nt*nemt
#        ldb = 1 #2*nrec
#        ldc = 1 #nt*nemt
#        mm = nt*nemt
#        kk = 2*nrec
#        scipy_blas.dgemv("N",  &mm, &kk, &alpha, &gff_v[0,0], &lda, &ext_field_v[0], &ldb, &beta, &ibc_mv_v[0], &ldc)
        ibc_mv = np.dot(gf_v,ext_field)
#        ibc2[...] = ibc2 + np.roll(ibc_mv.reshape((nemt,nt)).T, a, axis=0) / dx
        ibc3[:] = ibc2[a,:]
        ibc2[...] = ibc2 - np.roll(ibc_mv.reshape((nt,nemt), order='F'), a+1, axis=0)

        # INJECTION of EBCs
        # Monopole sources
        if ibctype == 1:
#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(nemt, schedule='dynamic'):
                for i in range(nemt):
                    p_v[semt_locs[i,0],semt_locs[i,2]] = p_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c1_v[semt_locs[i,0],semt_locs[i,2]]*(ibc3_v[i])

        # INJECTION of MPS
        # Monopole sources
        if ibctype == 1:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nemt, schedule='dynamic'):
                    if semt_locs[i,3] in {0, 2, 6, 8}:
                        p_v[semt_locs[i,0],semt_locs[i,2]] = p_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c1_v[semt_locs[i,0],semt_locs[i,2]]*(p_inj_v[a,i]) / dx
                    elif semt_locs[i,3] in {1, 3, 7, 9}:
                        p_v[semt_locs[i,0],semt_locs[i,2]] = p_v[semt_locs[i,0],semt_locs[i,2]] + 2 * c1_v[semt_locs[i,0],semt_locs[i,2]]*(p_inj_v[a,i]) / dx

        # Save snapshots
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nx, schedule='dynamic'):
                for j in range(nz):
                    snap_p_v[a,i,j] = p_v[i,j]
                    snap_vx_v[a,i,j] = vx_v[i,j]
                    snap_vz_v[a,i,j] = vz_v[i,j]

        # Save receiver wave fields
#        with nogil, parallel(num_threads=num_threads):
#            for i in prange(nrec, schedule='dynamic'):
#                f = rec_loc[i,3]
#                j = rec_loc[i,0]
#                js = face_v[f,0]
#                k = rec_loc[i,2]
#                ks = face_v[f,2]
#                rec_p_v[a,i] = 0.5 * (p_v[j,k] + p_v[j+js,k+ks])
#                if f == 0 or f == 1 or f == 6 or f == 7:
#                    rec_vx_v[a,i] = 0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
#                elif f == 2 or f == 3 or f == 8 or f == 9:
#                    rec_vx_v[a,i] = 0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])

    end = timer()
    print(end - start, 's')

#    return snap_p_v, snap_vx_v, snap_vz_v, ibc_v, ibc2, ext_field_out_v, p_mv, p_mv2, p_mv3
    return snap_p, snap_vx, snap_vz
#    if snap and not(gather_vb):
#        return rec_p, rec_vx, rec_vz, snap_p, snap_vx, snap_vz
#    elif not(snap) and not(gather_vb):
#        return rec_p, rec_vx, rec_vz
#    elif not(snap) and gather_vb:
#        return rec_p, rec_vx
#    else:
#        print('Something went wrong!')
#        return 0
