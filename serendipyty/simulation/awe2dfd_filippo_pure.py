# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:58:34 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

# %% Full model run on 2D staggered grid
# O(2,2)

from __future__ import division
from __future__ import print_function
from timeit import default_timer as timer
import numpy as np
cimport numpy as np

cimport cython
from cython.parallel cimport prange, parallel

import generate_pml_coeff
#reload(generate_pml_coeff)

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calc_full(double[:,::1] c, double[:,::1] rho, DTYPE_t dx,
              size_t[::1] src_loc, double[::1] wav_src, DTYPE_t fc, DTYPE_t dt, sourcetypestring,
              outparam,
              int pml, int npml, int num_threads, int verbose=0):

    cdef size_t a, i, j, k, l, js, ks, f

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
    cdef size_t u0 = np.uint(0)
    cdef size_t u1 = np.uint(1)
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

    cdef size_t[::1] src_loc_v = np.array(src_loc)

    cdef int nx = c.shape[0]
    cdef int nz = c.shape[1]
    cdef int nt = wav_src.shape[0]
    cdef int nrec #= rec_loc.shape[0]

    # Time parameters for outputs
    cdef size_t[::1] aa = np.zeros((len(outparam)), dtype=np.uint)
    cdef size_t[::1] step = np.ones((len(outparam)), dtype=np.uint)
    cdef size_t[::1] nts = np.ones((len(outparam)), dtype=np.uint)*nt
    cdef size_t aaa = 0
    cdef size_t stepa = 1
    cdef size_t ntsa = nt

    cdef DTYPE_t dz = dx
    cdef DTYPE_t d2 = 1.0/(dx*dz)
    cdef DTYPE_t vp0 = c[1,1]

    # Initialize output wave fields
    outtypes = {'slice': None,
                'sub_volume_boundary': None,
                'shot_gather': None
                }
    for m in outtypes:
       outtypes[m] = [i for i, n in enumerate(outparam) if n['type'] == m]
    print(outtypes)

    k = 0
    for m in outtypes:
        for j in outtypes[m]:
            step[k] = outparam[j]['timestep_increment']
            nts[k] = int(nt/step[k])
            print('nts is: ', nts[k])
            k = k + 1

    stepa = step[0]
    ntsa = nts[0]
    print('ntsa is: ', ntsa)
    print('stepa is: ', stepa)

    cdef np.ndarray[DTYPE_t, ndim=4, mode='c'] snap
    cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] svb

    cdef size_t[:,::1] srec_locs

    if len(outtypes['slice']) > 0:
#        print('# of slice: ', len(outtypes['slice']))
        m = outtypes['slice'][0]
#        step[m] = outparam[m]['timestep_increment']
#        nts[m] = int(nt/stepa)
        snap = np.zeros((3, ntsa, nx, nz), dtype=DTYPE)
#        print('nts is: ', nts[m])
    nsvb = len(outtypes['sub_volume_boundary'])
    # IMPORTANT
    # Below, I assume that all the svb outputs are recorded along
    # the same srec_locs locations
    if nsvb > 0:
#        print('# of svb: ', len(outtypes['sub_volume_boundary']))
        m = outtypes['sub_volume_boundary'][0]
        nrec = outparam[m]['receiver_locations'].shape[0]
        svb = np.zeros((nsvb, ntsa, nrec), dtype=DTYPE)
        srec_locs = outparam[m]['receiver_locations']

    # Staggering
    cdef double[::1] stag_v = np.zeros((nsvb), dtype=DTYPE)
    for l, m in enumerate(outtypes['sub_volume_boundary']):
        if outparam[m]['stagger_on_sub_volume'] == True:
            stag_v[l] = 1.0

    # Attributes
    cdef size_t[::1] attribute_v = np.zeros((nsvb), dtype=np.uint)
    for l, m in enumerate(outtypes['sub_volume_boundary']):
        if outparam[m]['attribute'] == 'p':
            attribute_v[l] = u0
        elif outparam[m]['attribute'] == 'vn':
            attribute_v[l] = u1

    #if pml < 4:
    #
    #    [a_x, b_x, K_x, a_x_half, b_x_half, K_x_half, a_z, b_z, K_z, a_z_half, b_z_half, K_z_half] = ...
    #        generate_pml_coeff_free(nx_pml,nz_pml,npml,fc,dx,dx,dt,c(1,1))
    #    c_pml = zeros(nx_pml,nz_pml)
    #    c_pml(npml+1:end-npml,1:nz) = c
    #    c_pml(1:npml,1:nz) = repmat(c(1,:),npml,1)
    #    c_pml(end-npml+1:end,1:nz) = repmat(c(end,:),npml,1)
    #    c_pml(:,nz+1:end) = c(1,nz)
    #
    #    rho_pml = zeros(nx_pml,nz_pml)
    #    rho_pml(npml+1:end-npml,1:nz) = rho
    #    rho_pml(1:npml,1:nz) = repmat(rho(1,:),npml,1)
    #    rho_pml(end-npml+1:end,1:nz) = repmat(rho(end,:),npml,1)
    #    rho_pml(:,nz+1:end) = rho(1,nz)
    #
    #    src_loc[0] = src_loc[0] + npml
    #

    cdef int nx_pml, nz_pml, xpad, zpad

    if pml == 3:
        # Free surface on top
        nx_pml, nz_pml = nx + 2*npml, nz + 1*npml
        xpad = npml
        zpad = 0
        src_loc_v[0] = src_loc_v[0] + npml
    elif pml == 4:
        nx_pml, nz_pml = nx + 2*npml, nz + 2*npml
        xpad = npml
        zpad = npml
        src_loc_v[0], src_loc_v[2] = src_loc_v[0] + npml, src_loc_v[2] + npml

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] c_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rho_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

    if pml == 3:
        c_pml = np.pad(c, ((npml, npml), (0, npml)), 'edge')
        rho_pml = np.pad(rho, ((npml, npml), (0, npml)), 'edge')
    elif pml == 4:
        c_pml = np.pad(c, ((npml, npml), (npml, npml)), 'edge')
        rho_pml = np.pad(rho, ((npml, npml), (npml, npml)), 'edge')

    # Initialize all the typed memoryviews
    cdef double[:,::1] c1_v = dt*rho_pml*(c_pml**2)
    cdef double[:,::1] c2_v = dt/rho_pml
    cdef double[:,::1] diffop_x_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
    cdef double[:,::1] diffop_z_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
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

    cdef double[:,:,:,::1] snap_v
    if len(outtypes['slice']) > 0:
        snap_v = snap

    cdef double[:,:,::1] svb_v
    if len(outtypes['sub_volume_boundary']) > 0:
        svb_v = svb

    # Compute PML coefficients
    a_x_v, b_x_v, K_x_v, a_x_half_v, b_x_half_v, K_x_half_v, a_z_v, b_z_v, K_z_v, a_z_half_v, b_z_half_v, K_z_half_v = \
        generate_pml_coeff.generate_pml_coeff(nx_pml, nz_pml, pml, npml, fc, dx, dz, dt, vp0)

    # Main loop
#    start = timer()
    for a in range(nt):

        if a%100 == 0 and verbose:
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
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nx_pml-1, schedule='dynamic'):
                for j in range(nz_pml):
                    diffop_p_x_v[i,j] = (p_v[i+1,j] - p_v[i,j]) / dx
                    px_mem_v[i,j] = b_x_half_v[i,j] * px_mem_v[i,j] + a_x_half_v[i,j] * diffop_p_x_v[i,j]
                    diffop_p_x_v[i,j] = diffop_p_x_v[i,j] / K_x_half_v[i,j] + px_mem_v[i,j]
                    vx_v[i,j] = vx_v[i,j] + c2_v[i,j] * diffop_p_x_v[i,j]

        with nogil, parallel(num_threads=num_threads):
            for i in prange(nx_pml, schedule='dynamic'):
                for j in range(nz_pml-1):
                    diffop_p_z_v[i,j] = (p_v[i,j+1] - p_v[i,j]) / dx
                    pz_mem_v[i,j] = b_z_half_v[i,j] * pz_mem_v[i,j] + a_z_half_v[i,j] * diffop_p_z_v[i,j]
                    diffop_p_z_v[i,j] = diffop_p_z_v[i,j] / K_z_half_v[i,j] + pz_mem_v[i,j]
                    vz_v[i,j] = vz_v[i,j] + c2_v[i,j] * diffop_p_z_v[i,j]

        # Update p
        with nogil, parallel(num_threads=num_threads):
            for i in prange(1,nx_pml-1, schedule='dynamic'):
            #for i in range(nx_pml-1):
                for j in range(1,nz_pml-1):
                    diffop_x_v[i,j] = (vx_v[i,j] - vx_v[i-1,j]) / dx
                    vx_mem_v[i,j+1] = b_x_v[i,j+1] * vx_mem_v[i,j+1] + a_x_v[i,j+1] * diffop_x_v[i,j]
                    diffop_x_v[i,j] = diffop_x_v[i,j] / K_x_v[i,j+1] + vx_mem_v[i,j+1]
                    diffop_z_v[i,j] = (vz_v[i,j] - vz_v[i,j-1]) / dx
                    vz_mem_v[i+1,j] = b_z_v[i+1,j] * vz_mem_v[i+1,j] + a_z_v[i+1,j] * diffop_z_v[i,j]
                    diffop_z_v[i,j] = diffop_z_v[i,j] / K_z_v[i+1,j] + vz_mem_v[i+1,j]
                    p_v[i,j] = p_v[i,j] + c1_v[i,j] * (diffop_x_v[i,j] + diffop_z_v[i,j])

###############################################################################
# Apply free surface boundary condition
###############################################################################

        # Free surface: mirror p at top
        if pml == 3:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nx_pml, schedule='dynamic'):
                    p_v[i,0] = -p_v[i,1]

###############################################################################
# Outputs
###############################################################################

        if a%stepa == 0 and aaa < ntsa:
            for m in outtypes['slice']:
                # Save snapshots
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(nx, schedule='dynamic'):
                        for j in range(nz):
                            snap_v[0,aaa,i,j] = p_v[i+xpad,j+zpad]
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(nx, schedule='dynamic'):
                        for j in range(nz):
                            snap_v[1,aaa,i,j] = vx_v[i+xpad,j+zpad]
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(nx, schedule='dynamic'):
                        for j in range(nz):
                            snap_v[2,aaa,i,j] = vz_v[i+xpad,j+zpad]
#            print(aa)

            for l, m in enumerate(outtypes['sub_volume_boundary']):
                with nogil, parallel(num_threads=num_threads):
                    for i in prange(nrec, schedule='dynamic'):
                        f = srec_locs[i,3]
                        j = srec_locs[i,0] + xpad
                        js = face_v[f,0]
                        k = srec_locs[i,2] + zpad
                        ks = face_v[f,2]

                        if attribute_v[l] == u0:
                            svb_v[l,aaa,i] = +0.5 * (p_v[j,k] + stag_v[l] * p_v[j+js,k+ks])
                        elif attribute_v[l] == u1:
                            if f in {0, 6}:
                                svb_v[l,aaa,i] = -0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
                            elif f in {1, 7}:
                                svb_v[l,aaa,i] = +0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
                            elif f in {2, 8}:
                                svb_v[l,aaa,i] = -0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])
                            elif f in {3, 9}:
                                svb_v[l,aaa,i] = +0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])
            aaa = aaa + 1

#        # Save receiver wave fields
#        if not(gather_vb):
#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(nrec, schedule='dynamic'):
#                    j = rec_loc[i,0] + xpad
#                    k = rec_loc[i,2] + zpad
#                    rec_p_v[a,i] = p_v[j, k]
#                    rec_vx_v[a,i] = vx_v[j, k]
#                    rec_vz_v[a,i] = vz_v[j, k]
#        else:
#            with nogil, parallel(num_threads=num_threads):
#                for i in prange(nrec, schedule='dynamic'):
#                    f = rec_loc[i,3]
#                    j = rec_loc[i,0] + xpad
#                    js = face_v[f,0]
#                    k = rec_loc[i,2] + zpad
#                    ks = face_v[f,2]
#                    rec_p_v[a,i] = +0.5 * (p_v[j,k] + p_v[j+js,k+ks])
#                    if f in {0, 6}:
#                        rec_vx_v[a,i] = -0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
#                    elif f in {1, 7}:
#                        rec_vx_v[a,i] = +0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
#                    elif f in {2, 8}:
#                        rec_vx_v[a,i] = -0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])
#                    elif f in {3, 9}:
#                        rec_vx_v[a,i] = +0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])

#    end = timer()
#    print(end - start, 's')

    outputs = {}
    if len(outtypes['slice']) > 0:
        outputs['slice'] = snap
    if len(outtypes['sub_volume_boundary']) > 0:
        outputs['sub_volume_boundary'] = svb
#    if len(outtypes['slice']) > 0:
#        outputs['slice'] = snap
#    print(type(svb))
#    print(type(outputs))
    return outputs

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
def calc_full_old(double[:,::1] c, double[:,::1] rho, DTYPE_t dx,
              size_t[::1] src_loc, double[::1] wav_src, DTYPE_t fc, DTYPE_t dt, sourcetypestring,
              size_t[:,::1] rec_loc, int snap, int gather_vb,
              int pml, int npml, int num_threads):

    cdef size_t a, i, j, k, js, ks, f

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

    cdef size_t[::1] src_loc_v = np.array(src_loc)

    cdef int nx = c.shape[0]
    cdef int nz = c.shape[1]
    cdef int nt = wav_src.shape[0]
    cdef int nrec = rec_loc.shape[0]

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

    #if pml < 4:
    #
    #    [a_x, b_x, K_x, a_x_half, b_x_half, K_x_half, a_z, b_z, K_z, a_z_half, b_z_half, K_z_half] = ...
    #        generate_pml_coeff_free(nx_pml,nz_pml,npml,fc,dx,dx,dt,c(1,1))
    #    c_pml = zeros(nx_pml,nz_pml)
    #    c_pml(npml+1:end-npml,1:nz) = c
    #    c_pml(1:npml,1:nz) = repmat(c(1,:),npml,1)
    #    c_pml(end-npml+1:end,1:nz) = repmat(c(end,:),npml,1)
    #    c_pml(:,nz+1:end) = c(1,nz)
    #
    #    rho_pml = zeros(nx_pml,nz_pml)
    #    rho_pml(npml+1:end-npml,1:nz) = rho
    #    rho_pml(1:npml,1:nz) = repmat(rho(1,:),npml,1)
    #    rho_pml(end-npml+1:end,1:nz) = repmat(rho(end,:),npml,1)
    #    rho_pml(:,nz+1:end) = rho(1,nz)
    #
    #    src_loc[0] = src_loc[0] + npml
    #

    cdef int nx_pml, nz_pml, xpad, zpad

    if pml == 3:
        # Free surface on top
        nx_pml, nz_pml = nx + 2*npml, nz + 1*npml
        xpad = npml
        zpad = 0
        src_loc_v[0] = src_loc_v[0] + npml
    elif pml == 4:
        nx_pml, nz_pml = nx + 2*npml, nz + 2*npml
        xpad = npml
        zpad = npml
        src_loc_v[0], src_loc_v[2] = src_loc_v[0] + npml, src_loc_v[2] + npml

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] c_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rho_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

    if pml == 3:
        c_pml = np.pad(c, ((npml, npml), (0, npml)), 'edge')
        rho_pml = np.pad(rho, ((npml, npml), (0, npml)), 'edge')
    elif pml == 4:
        c_pml = np.pad(c, ((npml, npml), (npml, npml)), 'edge')
        rho_pml = np.pad(rho, ((npml, npml), (npml, npml)), 'edge')

    # Initialize all the typed memoryviews
    cdef double[:,::1] c1_v = dt*rho_pml*(c_pml**2)
    cdef double[:,::1] c2_v = dt/rho_pml
    cdef double[:,::1] diffop_x_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
    cdef double[:,::1] diffop_z_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
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

    # Compute PML coefficients
    a_x_v, b_x_v, K_x_v, a_x_half_v, b_x_half_v, K_x_half_v, a_z_v, b_z_v, K_z_v, a_z_half_v, b_z_half_v, K_z_half_v = \
        generate_pml_coeff.generate_pml_coeff(nx_pml, nz_pml, pml, npml, fc, dx, dz, dt, vp0)

    # Main loop
#    start = timer()
    for a in range(nt):

#        if a%10000 == 0:
#            print('FULL model, timestepp = ', a, 1)

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
        with nogil, parallel(num_threads=num_threads):
            for i in prange(nx_pml-1, schedule='dynamic'):
                for j in range(nz_pml):
                    diffop_p_x_v[i,j] = (p_v[i+1,j] - p_v[i,j]) / dx
                    px_mem_v[i,j] = b_x_half_v[i,j] * px_mem_v[i,j] + a_x_half_v[i,j] * diffop_p_x_v[i,j]
                    diffop_p_x_v[i,j] = diffop_p_x_v[i,j] / K_x_half_v[i,j] + px_mem_v[i,j]
                    vx_v[i,j] = vx_v[i,j] + c2_v[i,j] * diffop_p_x_v[i,j]

        with nogil, parallel(num_threads=num_threads):
            for i in prange(nx_pml, schedule='dynamic'):
                for j in range(nz_pml-1):
                    diffop_p_z_v[i,j] = (p_v[i,j+1] - p_v[i,j]) / dx
                    pz_mem_v[i,j] = b_z_half_v[i,j] * pz_mem_v[i,j] + a_z_half_v[i,j] * diffop_p_z_v[i,j]
                    diffop_p_z_v[i,j] = diffop_p_z_v[i,j] / K_z_half_v[i,j] + pz_mem_v[i,j]
                    vz_v[i,j] = vz_v[i,j] + c2_v[i,j] * diffop_p_z_v[i,j]

        # Update p
        with nogil, parallel(num_threads=num_threads):
            for i in prange(1,nx_pml-1, schedule='dynamic'):
            #for i in range(nx_pml-1):
                for j in range(1,nz_pml-1):
                    diffop_x_v[i,j] = (vx_v[i,j] - vx_v[i-1,j]) / dx
                    vx_mem_v[i,j+1] = b_x_v[i,j+1] * vx_mem_v[i,j+1] + a_x_v[i,j+1] * diffop_x_v[i,j]
                    diffop_x_v[i,j] = diffop_x_v[i,j] / K_x_v[i,j+1] + vx_mem_v[i,j+1]
                    diffop_z_v[i,j] = (vz_v[i,j] - vz_v[i,j-1]) / dx
                    vz_mem_v[i+1,j] = b_z_v[i+1,j] * vz_mem_v[i+1,j] + a_z_v[i+1,j] * diffop_z_v[i,j]
                    diffop_z_v[i,j] = diffop_z_v[i,j] / K_z_v[i+1,j] + vz_mem_v[i+1,j]
                    p_v[i,j] = p_v[i,j] + c1_v[i,j] * (diffop_x_v[i,j] + diffop_z_v[i,j])

###############################################################################
# Apply free surface boundary condition
###############################################################################

        # Free surface: mirror p at top
        if pml == 3:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nx_pml, schedule='dynamic'):
                    p_v[i,0] = -p_v[i,1]

###############################################################################
# Outputs
###############################################################################

        # Save snapshots
        if pml == 3 and snap:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nx, schedule='dynamic'):
                    for j in range(nz):
                        snap_p_v[a,i,j] = p_v[i+npml,j]
                        snap_vx_v[a,i,j] = vx_v[i+npml,j]
                        snap_vz_v[a,i,j] = vz_v[i+npml,j]
        elif pml == 4 and snap:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nx, schedule='dynamic'):
                    for j in range(nz):
                        snap_p_v[a,i,j] = p_v[i+npml,j+npml]
                        snap_vx_v[a,i,j] = vx_v[i+npml,j+npml]
                        snap_vz_v[a,i,j] = vz_v[i+npml,j+npml]

        # Save receiver wave fields
        if not(gather_vb):
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nrec, schedule='dynamic'):
                    j = rec_loc[i,0] + xpad
                    k = rec_loc[i,2] + zpad
                    rec_p_v[a,i] = p_v[j, k]
                    rec_vx_v[a,i] = vx_v[j, k]
                    rec_vz_v[a,i] = vz_v[j, k]
        else:
            with nogil, parallel(num_threads=num_threads):
                for i in prange(nrec, schedule='dynamic'):
                    f = rec_loc[i,3]
                    j = rec_loc[i,0] + xpad
                    js = face_v[f,0]
                    k = rec_loc[i,2] + zpad
                    ks = face_v[f,2]
                    rec_p_v[a,i] = +0.5 * (p_v[j,k] + p_v[j+js,k+ks])
                    if f in {0, 6}:
                        rec_vx_v[a,i] = -0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
                    elif f in {1, 7}:
                        rec_vx_v[a,i] = +0.5 * (vx_v[j,k] + vx_v[j-js,k-ks])
                    elif f in {2, 8}:
                        rec_vx_v[a,i] = -0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])
                    elif f in {3, 9}:
                        rec_vx_v[a,i] = +0.5 * (vz_v[j,k] + vz_v[j-js,k-ks])

#    end = timer()
#    print(end - start, 's')

    if snap and not(gather_vb):
        return rec_p, rec_vx, rec_vz, snap_p, snap_vx, snap_vz
    elif not(snap) and not(gather_vb):
        return rec_p, rec_vx, rec_vz
    elif not(snap) and gather_vb:
        return rec_p, rec_vx
    else:
        print('Something went wrong!')
        return 0
