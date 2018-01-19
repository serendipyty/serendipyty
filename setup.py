

"""Serendipyty: Seismic Inversion Toolbox in Python
Serendipyty is a collection of geophysical algorithms for Python.
"""


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

ext_modules = [
	Extension(name="awe2dfd", sources=["serendipyty/seismic/modelling/awe2dfd_filippo.pyx"],
	  extra_compile_args = ["-O3", "-ffast-math", "-mavx", "-fopenmp"],
	  extra_link_args=['-fopenmp'],
      include_dirs = [np.get_include()]
	  ),
	Extension(name="generate_pml_coeff", sources=["serendipyty/seismic/modelling/generate_pml_coeff.pyx"],
	  extra_compile_args = ["-O3", "-ffast-math", "-mavx", "-fopenmp"],
	  extra_link_args=['-fopenmp'],
      include_dirs = [np.get_include()]
	  ),
	Extension(name="ebc", sources=["serendipyty/seismic/modelling/ebc_filippo.pyx"],
	  extra_compile_args = ["-O3", "-ffast-math", "-mavx", "-fopenmp"],
	  extra_link_args=['-fopenmp'],
      include_dirs = [np.get_include()]
	  ),
	]
	  #include_dirs = [numpy.get_include()]
#	Extension(name="generate_pml_coeff", sources=["generate_pml_coeff.pyx"], extra_compile_args = ["-O3", "-fopenmp"]),
#	]


# Setup data inclusion
package_data = {}
# {'modeling':['../cylinalg/cylinalg/*.pxd']},

setup(
    name = "serendipyty",
    packages = ['modelling'],
    package_data = package_data,
    include_dirs=['modelling'],
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
    install_requires = ['python',
                       ],
    author = "Filippo Broggini",
    author_email = "filippo.broggini@erdw.ethz.ch",
    description = "Serendipyty: a Python library for learning and teaching Geophysics",
    license = "GPLv3",
    keywords = "seismic modelling imaging",
    url = "https://serendipyty.github.io/",
    download_url = "https://github.com/serendipyty/serendipyty/",
    )
