# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:58:34 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

# %% Full model run on 2D staggered grid
# O(2,2)

from timeit import default_timer as timer
import numpy as np
cimport numpy as np

# Cython imports
cimport cython
from cython.parallel cimport prange, parallel
cimport serendipyty.seismic.modelling.generate_pml_coeff as generate_pml_coeff

# Set floating point precision
ctypedef double MVTYPE
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def forward(model, src, outparam, bc, hpc=None, int verbose=0):
    r"""Acoustic wave equation forward modeling.

    Finite-difference time-domain solution of the 2D acoustic wave equation
    implemented using Cython.

    Parameters
    ----------
    model : BaseModel
        Model class.
    src : BaseSource
        Source class.
    outparam: list
        List of outputs.
    bc: BaseBc
        Boundary conditions class.
    hpc: BaseHpc
        HPC class.
    verbose: int
        Set to 1 output more information.

    Returns
    -------
    outputs: dict
        Dictionary containing the outputs required by the input outparam list.
    """

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

    #
    cdef size_t u0 = 0 #np.uint(0)
    cdef size_t u1 = 1 #np.uint(1)

###############################################################################
# Source and time parameters
###############################################################################
    cdef int sourcetype
    if type(src.mode) is str:
        if src.mode == 'q':
            sourcetype = 0
        elif src.mode == 'fx':
            sourcetype = 1
        elif src.mode == 'fz':
            sourcetype = 3
    else:
        if src.mode in {0, 1, 6, 7}:
            sourcetype = 1 # fx
        elif src.mode in {2, 3, 8, 9}:
            sourcetype = 3 # fz

    cdef MVTYPE[::1] wavelet_v = src.wavelet.wavelet
    cdef size_t[::1] src_loc_v = src.loc
    cdef int nt = src.nt
    cdef DTYPE_t dt = src.wavelet.dt
    cdef DTYPE_t fc = src.wavelet.fc

    # Time parameters for outputs
    cdef size_t[::1] aa = np.zeros((len(outparam)), dtype=np.uint)
    cdef size_t[::1] step = np.ones((len(outparam)), dtype=np.uint)
    cdef size_t[::1] nts = np.ones((len(outparam)), dtype=np.uint)*nt
    cdef size_t aaa = 0
    cdef size_t stepa = 1
    cdef size_t ntsa = nt

###############################################################################
# Model and grid parameters
###############################################################################
    cdef int nx = model.n[0]
    cdef int nz = model.n[2]

    cdef int nrec #= rec_loc.shape[0]

    cdef DTYPE_t dx = model.dx
    cdef DTYPE_t dz = model.dx
    cdef DTYPE_t d2 = 1.0/(model.dx*model.dz)
    cdef DTYPE_t vp0 = model.model[1,1,0]

###############################################################################
# HPC parameters
###############################################################################
    cdef int num_threads = hpc.omp_num_threads

###############################################################################
# Output parameters
###############################################################################
    # Initialize output wave fields
    outtypes = {'slice': None,
                'sub_volume_boundary': None,
                'shot_gather': None
                }
    for m in outtypes:
       outtypes[m] = [i for i, n in enumerate(outparam) if n['type'] == m]
    print('The outputs are: {}'.format(outtypes))

    k = 0
    for m in outtypes:
        for j in outtypes[m]:
            step[k] = outparam[j]['timestep_increment']
            nts[k] = int(nt/step[k])
            # print('nts is: ', nts[k])
            k = k + 1

    stepa = step[0]
    ntsa = nts[0]
    # print('ntsa is: ', ntsa)
    # print('stepa is: ', stepa)

    # Output arrays for snapshots and sub-volume boundaries
    # cdef np.ndarray[DTYPE_t, ndim=4, mode='c'] snap
    # cdef np.ndarray[DTYPE_t, ndim=3, mode='c'] svb

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
        #print('Before: {}'.format(type(srec_locs)))
        srec_locs = outparam[m]['receiver_locations']
        #print('After: {}'.format(type(srec_locs)))

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

###############################################################################
# Boundary conditions (BCs) parameters
###############################################################################
    cdef int nx_pml, nz_pml, xpad, zpad

    # Free surface on top
    if bc.freesurface:
        nx_pml, nz_pml = nx + 2*bc.npml, nz + 1*bc.npml
        xpad = bc.npml
        zpad = 0
        src_loc_v[0] = src_loc_v[0] + bc.npml
    else:
        nx_pml, nz_pml = nx + 2*bc.npml, nz + 2*bc.npml
        xpad = bc.npml
        zpad = bc.npml
        src_loc_v[0], src_loc_v[2] = src_loc_v[0] + bc.npml, src_loc_v[2] + bc.npml

    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] vp_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2, mode='c'] rho_pml = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

    # Free surface on top
    if bc.freesurface:
        vp_pml = np.pad(model.model[..., 0], ((bc.npml, bc.npml), (0, bc.npml)), 'edge')
        rho_pml = np.pad(model.model[..., 1], ((bc.npml, bc.npml), (0, bc.npml)), 'edge')
    else:
        vp_pml = np.pad(model.model[..., 0], ((bc.npml, bc.npml), (bc.npml, bc.npml)), 'edge')
        rho_pml = np.pad(model.model[..., 1], ((bc.npml, bc.npml), (bc.npml, bc.npml)), 'edge')

###############################################################################
# Initialize all the typed memoryviews
###############################################################################
    cdef MVTYPE[:,::1] c1_v = dt*rho_pml*(vp_pml**2)
    cdef MVTYPE[:,::1] c2_v = dt/rho_pml
    cdef MVTYPE[:,::1] diffop_x_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
    cdef MVTYPE[:,::1] diffop_z_v = np.zeros((nx_pml-1, nz_pml-1), dtype=DTYPE)
    cdef MVTYPE[:,::1] diffop_p_x_v = np.zeros((nx_pml-1, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] diffop_p_z_v = np.zeros((nx_pml, nz_pml-1), dtype=DTYPE)
    cdef MVTYPE[:,::1] p_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] px_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] pz_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] vx_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] vx_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] vz_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] vz_mem_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] a_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] b_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] K_x_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] a_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] b_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] K_z_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] a_x_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] b_x_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] K_x_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] a_z_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] b_z_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)
    cdef MVTYPE[:,::1] K_z_half_v = np.zeros((nx_pml, nz_pml), dtype=DTYPE)

    cdef MVTYPE[:,:,:,::1] snap_v
    if len(outtypes['slice']) > 0:
        snap_v = snap

    cdef MVTYPE[:,:,::1] svb_v
    if len(outtypes['sub_volume_boundary']) > 0:
        svb_v = svb

###############################################################################
# Compute PML coefficients
###############################################################################
    a_x_v, b_x_v, K_x_v, a_x_half_v, b_x_half_v, K_x_half_v, a_z_v, b_z_v, K_z_v, a_z_half_v, b_z_half_v, K_z_half_v = \
        generate_pml_coeff.generate_pml_coeff(nx_pml, nz_pml, bc.freesurface, bc.npml, fc, dx, dz, dt, vp0)

###############################################################################
# Main loop over time
###############################################################################
#    start = timer()
    for a in range(nt):

        if a%100 == 0 and verbose:
            print('FULL model, timestepp = ', a, 1)

        # Inject velocity source
        if sourcetype == 1:
            vx_v[src_loc_v[0],src_loc_v[2]] = vx_v[src_loc_v[0],src_loc_v[2]] + c2_v[src_loc_v[0],src_loc_v[2]]*0.5*wavelet_v[a]*d2
            vx_v[src_loc_v[0]-1,src_loc_v[2]] = vx_v[src_loc_v[0]-1,src_loc_v[2]] + c2_v[src_loc_v[0]-1,src_loc_v[2]]*0.5*wavelet_v[a]*d2
        elif sourcetype == 3:
            vz_v[src_loc_v[0],src_loc_v[2]] = vz_v[src_loc_v[0],src_loc_v[2]] + c2_v[src_loc_v[0],src_loc_v[2]]*0.5*wavelet_v[a]*d2
            vz_v[src_loc_v[0],src_loc_v[2]-1] = vz_v[src_loc_v[0],src_loc_v[2]-1] + c2_v[src_loc_v[0],src_loc_v[2]-1]*0.5*wavelet_v[a]*d2

        # Inject pressure source
        if sourcetype == 0:
            p_v[src_loc_v[0],src_loc_v[2]] = p_v[src_loc_v[0],src_loc_v[2]] + c1_v[src_loc_v[0],src_loc_v[2]]*wavelet_v[a]*d2

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
        if bc.freesurface:
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
