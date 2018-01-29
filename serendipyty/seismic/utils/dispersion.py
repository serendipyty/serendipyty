# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7 11:35:44 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from timeit import default_timer as timer

DTYPE = np.float64

def itdt(trace):
    """
    ITDT Inverse Time Dispersion Transform
    This function artificially REMOVES time dispersion from a vector of samples
    in time. It models the phase shift error. It uses the fact that every
    sample equals 1 time step (assuming equidistant sampling), removing the dt dependency.
    f    [VEC]            trace with dispersion
    itdt [VEC] (n x 1)    trace without dispersion
    """
    # Prepare data
    nt = trace.shape[0]
    nt2 = nt*2.0
    f = np.zeros((nt2,))
    f[:nt] = trace

    print('Shape of f is ', f.shape)

    # The phase shift function
    omega = 2.0*np.pi*np.arange(nt2)/nt2 + 0.0j

    omax = np.where(omega > 2.0)[0][0]

    fn = 2.0*np.arcsin(omega[0:omax]/2.0)

    a = -1.0j * np.arange(nt2)

#    b = np.outer(a, fn).T
    start = timer()
    b = np.outer(fn, a)
    end = timer()
    print('np.outer: ', end - start, 's')
    print('Shape of b is ', b.shape)

    print(b.flags)

    start = timer()
    c = np.exp(b)
    end = timer()
    print('np.exp: ', end - start, 's')
    print('Shape of c is ', c.shape)

    # Time -> Freq -> Time transforms
    # Apply ITDT in an altered DFT
    start = timer()
    IFTDTuf = np.dot(c, f)
    end = timer()
    print('np.dot: ', end - start, 's')
    print('Shape of IFTDTuf is ', IFTDTuf.shape)

    # Bring back to the time domain
    d = np.pad(IFTDTuf, (0,int(nt2-omax)), 'constant')
    start = timer()
    tracenew  = np.fft.irfft(np.pad(IFTDTuf, (0,int(nt2-omax)), 'constant'), int(nt2))
    end = timer()
    print('np.fft.irfft: ', end - start, 's')
    print('Shape of tracenew is ', tracenew.shape)
    tracenew  = tracenew[0:nt]

    return tracenew, IFTDTuf

def itdt2(trace):
    """
    ITDT Inverse Time Dispersion Transform
    This function artificially REMOVES time dispersion from a vector of samples
    in time. It models the phase shift error. It uses the fact that every
    sample equals 1 time step (assuming equidistant sampling), removing the dt dependency.
    f    [VEC]            trace with dispersion
    itdt [VEC] (n x 1)    trace without dispersion
    """
    # Prepare data
    nr, nt = trace.shape
    nt2 = int(nt*2)
    f = np.zeros((nr, nt2))
    f[...] = np.pad(trace, ((0, 0), (0, nt)), 'constant')

    print('Shape of f is ', f.shape)

    # The phase shift function
    omega = 2.0*np.pi*np.arange(nt2)/nt2 + 0.0j

    omax = int(np.where(omega > 2.0)[0][0])

    fn = 2.0*np.arcsin(omega[0:omax]/2.0)

    a = -1.0j * np.arange(nt2)

#    b = np.outer(a, fn).T
    start = timer()
    b = np.outer(a, fn)
    end = timer()
    print('np.outer: ', end - start, 's')
    print('Shape of b is ', b.shape)

    print(b.flags)

    start = timer()
    c = np.exp(b)
    end = timer()
    print('np.exp: ', end - start, 's')
    print('Shape of c is ', c.shape)

    # Time -> Freq -> Time transforms
    # Apply ITDT in an altered DFT
    start = timer()
    IFTDTuf = np.dot(f, c)
    end = timer()
    print('np.dot: ', end - start, 's')
    print('Shape of IFTDTuf is ', IFTDTuf.shape)

    # Bring back to the time domain
    d = np.pad(IFTDTuf, ((0, 0), (0, nt2-omax)), 'constant')
    start = timer()
    tracenew  = np.fft.irfft(d, nt2)
    end = timer()
    print('np.fft.irfft: ', end - start, 's')
    print('Shape of tracenew is ', tracenew.shape)
    tracenew  = tracenew[:, 0:nt]

    return tracenew, IFTDTuf


def itdt3(trace, oversample):
    """
    ITDT Inverse Time Dispersion Transform
    This function artificially REMOVES time dispersion from a vector of samples
    in time. It models the phase shift error. It uses the fact that every
    sample equals 1 time step (assuming equidistant sampling), removing the dt dependency.
    f    [VEC]            trace with dispersion
    itdt [VEC] (n x 1)    trace without dispersion
    """
    # Prepare data
    nr, nt = trace.shape
    nt2 = int(nt*2)
    f = np.zeros((nr, nt2))
    f[...] = np.pad(trace, ((0, 0), (0, nt)), 'constant')

    print('Shape of f is ', f.shape)

    # The phase shift function
    omega = 2.0*np.pi*np.arange(nt2)/nt2/oversample + 0.0j

    omax = int(np.where(omega < 2.0)[0][-1] + 1)

    fn = 2.0*np.arcsin(omega[0:omax]/2.0)

    a = -1.0j * np.arange(nt2)

#    b = np.outer(a, fn).T
    start = timer()
    b = np.outer(a, fn) * oversample
    end = timer()
    print('np.outer: ', end - start, 's')
    print('Shape of b is ', b.shape)

    print(b.flags)

    start = timer()
    c = np.exp(b)
    end = timer()
    print('np.exp: ', end - start, 's')
    print('Shape of c is ', c.shape)

    # Time -> Freq -> Time transforms
    # Apply ITDT in an altered DFT
    start = timer()
    IFTDTuf = np.dot(f, c)
    end = timer()
    print('np.dot: ', end - start, 's')
    print('Shape of IFTDTuf is ', IFTDTuf.shape)

    # Bring back to the time domain
    d = np.pad(IFTDTuf, ((0, 0), (0, nt2-omax)), 'constant')
    start = timer()
    tracenew  = np.fft.irfft(d, nt2)
    end = timer()
    print('np.fft.irfft: ', end - start, 's')
    print('Shape of tracenew is ', tracenew.shape)
    tracenew  = tracenew[:, 0:nt]

    return tracenew, IFTDTuf


def ftdt3(trace, oversample):
    """
    FTDT Forward Time Dispersion Transform
    This function artificially INTRODUCE time dispersion from a vector of samples
    in time. It models the phase shift error. It uses the fact that every
    sample equals 1 time step (assuming equidistant sampling), removing the dt dependency.
    f    [VEC]            trace with dispersion
    itdt [VEC] (n x 1)    trace without dispersion
    """
    # Prepare data
    nr, nt = trace.shape
    nt2 = int(nt*2)
    f = np.zeros((nr, nt2))
    f[...] = np.pad(trace, ((0, 0), (0, nt)), 'constant')

    print('Shape of f is ', f.shape)

    # The phase shift function
    omega = 2.0*np.pi*np.arange(nt2)/nt2/oversample + 0.0j

    omax = int(np.where(omega < np.pi)[0][-1] + 1)

    fn = 2.0*np.sin(omega[0:omax]/2.0)

    a = -1.0j * np.arange(nt2)

#    b = np.outer(a, fn).T
    start = timer()
    b = np.outer(a, fn) * oversample
    end = timer()
    print('np.outer: ', end - start, 's')
    print('Shape of b is ', b.shape)

    print(b.flags)

    start = timer()
    c = np.exp(b)
    end = timer()
    print('np.exp: ', end - start, 's')
    print('Shape of c is ', c.shape)

    # Time -> Freq -> Time transforms
    # Apply ITDT in an altered DFT
    start = timer()
    IFTDTuf = np.dot(f, c)
    end = timer()
    print('np.dot: ', end - start, 's')
    print('Shape of IFTDTuf is ', IFTDTuf.shape)

    # Bring back to the time domain
    d = np.pad(IFTDTuf, ((0, 0), (0, nt2-omax)), 'constant')
    start = timer()
    tracenew  = np.fft.irfft(d, nt2)
    end = timer()
    print('np.fft.irfft: ', end - start, 's')
    print('Shape of tracenew is ', tracenew.shape)
    tracenew  = tracenew[:, 0:nt]

    return tracenew, IFTDTuf