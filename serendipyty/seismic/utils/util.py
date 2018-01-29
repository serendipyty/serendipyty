# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 13:35:44 2016

@author: Filippo Broggini (ETH Zürich) - filippo.broggini@erdw.ethz.ch
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from timeit import default_timer as timer

DTYPE = np.float64

def ricker(fc, nt, dt):
    t_source = 1/fc
    t0 = t_source*1.5
    t = np.linspace(0,(nt-1)*dt,nt, dtype=DTYPE)
    tau = np.pi*(t-t0)/t0
    a = 4.0
    src = (1-a*tau*tau)*np.exp(-(a/2)*tau*tau)

    return src

def rickermh(fc, nt, dt):
    t_source = 1/fc
    t0 = t_source*1
    t = np.linspace(0,(nt-1)*dt,nt, dtype=DTYPE)
    tau = np.pi*(t-t0)/t0
    a = 2.0
    src = (1-a*tau*tau)*np.exp(-(a/2)*tau*tau)

    return src

def rectangle(**kwargs):
    """ Create rectangular surface """

    nfaces = 4

    locations = []

    origin = kwargs['origin']
    number_of_cells = kwargs['number_of_cells']
    cell_size = kwargs['cell_size']

    facedict = {0: (origin[0],
                    number_of_cells[2],
                    origin[2],
                    cell_size[2],
                    (0, 0, 1)),
                1: (origin[0] + (number_of_cells[0] - 1)*cell_size[0],
                    number_of_cells[2],
                    origin[2],
                    cell_size[2],
                    (0, 0, 1)),
                2: (origin[0],
                    number_of_cells[0],
                    origin[2],
                    cell_size[0],
                    (1, 0, 0)),
                3: (origin[0],
                    number_of_cells[0],
                    origin[2] + (number_of_cells[2] - 1)*cell_size[2],
                    cell_size[0],
                    (1, 0, 0))
                }

    for face in range(nfaces):
        for i in range(facedict[face][1]):
#                for j in range(number_of_cells[1]):
#                    y_emt = origin[1] + j*cell_size[1]
                if i == 0 or i == (facedict[face][1] - 1):
                    # Corner
                    flag = face + 6
                else:
                    # Edge (no corner)
                    flag = face
                locations.append((facedict[face][0] +
                                       facedict[face][4][0]*i*facedict[face][3],
                                       0,
                                       facedict[face][2] +
                                       facedict[face][4][2]*i*facedict[face][3],
                                       flag))
    return locations

def oneface(**kwargs):
    """ Create rectangular surface """

    faces = kwargs['faces']

    locations = []

    origin = kwargs['origin']
    number_of_cells = kwargs['number_of_cells']
    cell_size = kwargs['cell_size']

    facedict = {0: (origin[0],
                    number_of_cells[2],
                    origin[2],
                    cell_size[2],
                    (0, 0, 1)),
                1: (origin[0] + (number_of_cells[0] - 1)*cell_size[0],
                    number_of_cells[2],
                    origin[2],
                    cell_size[2],
                    (0, 0, 1)),
                2: (origin[0],
                    number_of_cells[0],
                    origin[2],
                    cell_size[0],
                    (1, 0, 0)),
                3: (origin[0],
                    number_of_cells[0],
                    origin[2] + (number_of_cells[2] - 1)*cell_size[2],
                    cell_size[0],
                    (1, 0, 0))
                }

    for face in faces:
        for i in range(facedict[face][1]):
#                for j in range(number_of_cells[1]):
#                    y_emt = origin[1] + j*cell_size[1]
#                if i == 0 or i == (facedict[face][1] - 1):
#                    # Corner
#                    flag = face + 6
#                else:
#                    # Edge (no corner)
#                    flag = face
            flag = face
            locations.append((facedict[face][0] +
                                   facedict[face][4][0]*i*facedict[face][3],
                                   0,
                                   facedict[face][2] +
                                   facedict[face][4][2]*i*facedict[face][3],
                                   flag))
    return locations
