# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:58:34 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

import numpy as np
cimport numpy as np
from cpython cimport bool

DTYPE = np.float64

ctypedef np.float64_t DTYPE_t

cdef generate_pml_coeff(int, int, bool, int, DTYPE_t, DTYPE_t, DTYPE_t, DTYPE_t, DTYPE_t)
