#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:10:52 2018

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

__all__ = ['BaseHpc']


class BaseHpc(object):
    r""" Base class for HPC parameters.

    This class defines the basic parameters for the HPC implementations
    used in the Serendipyty package.

    Parameters
    ----------
    omp_num_threads : int, optional
        Number of computational threads

    """

    def __init__(self, omp_num_threads=1):

        self.omp_num_threads = omp_num_threads
