# -*- coding: utf-8 -*-
#
"""setuptools-based setup.py template for Cython projects.

Main setup for the library.

Supports Python 3.4.

Usage as usual with setuptools:
    python setup.py build_ext # Build the extensions (as shared libraries)
    python setup.py build
    python setup.py install
    python setup.py sdist

For details, see
    http://setuptools.readthedocs.io/en/latest/setuptools.html#command-reference
or
    python setup.py --help
    python setup.py --help-commands
    python setup.py --help bdist_wheel  # or any command
"""

#from __future__ import division, print_function, absolute_import
import numpy as np
from sys import platform

try:
    # Python 3
    MyFileNotFoundError = FileNotFoundError
except:  # FileNotFoundError does not exist in Python 2.7
    # Python 2.7
    # - open() raises IOError
    # - remove() (not currently used here) raises OSError
    MyFileNotFoundError = (IOError, OSError)

#########################################################
# General config
#########################################################

# Name of the top-level package of your library.
#
# This is also the top level of its source tree, relative to the top-level project directory setup.py resides in.
#
libname="serendipyty"

# Choose build type.
#
build_type="optimized"
#build_type="debug"

# Short description for package list on PyPI
#
SHORTDESC="A Python library for learning and teaching Geophysics"

# Long description for package homepage on PyPI
#
DESC="""A Python library for learning and teaching Geophysics.

The focus of this template is on numerical scientific projects,
where a custom Cython extension (containing all-new code) can bring a large speedup.

Supports Python >= 3.4.
"""

# Set up data files for packaging.
#
# Directories (relative to the top-level directory where setup.py resides) in which to look for data files.
datadirs  = ("test",)

# File extensions to be considered as data files. (Literal, no wildcards.)
dataexts  = (".py",  ".pyx", ".pxd",  ".c", ".cpp", ".h",  ".sh",  ".lyx", ".tex", ".txt", ".pdf")

# Standard documentation to detect (and package if it exists).
standard_docs     = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]  # just the basename without file extension
standard_doc_exts = [".md", ".rst", ".txt", ""]  # commonly .md for GitHub projects, but other projects may use .rst or .txt (or even blank).

#########################################################
# Init
#########################################################

# check for Python 3.4 or later
# http://stackoverflow.com/questions/19534896/enforcing-python-version-in-setup-py
import sys
if sys.version_info < (3,4):
    sys.exit('Sorry, Python < 3.4 is not supported')

import os

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules.")

# This is required to automate cython compilation when installing with Pypi
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        #import numpy
        #self.include_dirs.append(numpy.get_include())

#########################################################
# Definitions
#########################################################

# Define our base set of compiler and linker flags.
#
# This is geared toward x86_64, see
#    https://gcc.gnu.org/onlinedocs/gcc-4.6.4/gcc/i386-and-x86_002d64-Options.html
#
# Customize these as needed.
#
# Note that -O3 may sometimes cause mysterious problems, so we limit ourselves to -O2.

if platform == 'darwin':
    os.environ["CC"] = "gcc-7"

# Modules involving numerical computations
extra_compile_args_math_optimized    = ['-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
extra_compile_args_math_debug        = ['-march=native', '-O0', '-g']
extra_link_args_math_optimized       = []
extra_link_args_math_debug           = []

# Modules that do not involve numerical computations
extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']
extra_link_args_nonmath_optimized    = []
extra_link_args_nonmath_debug        = []

# Additional flags to compile/link with OpenMP
openmp_compile_args = ['-fopenmp',]
openmp_link_args    = ['-fopenmp',]


#########################################################
# Helpers
#########################################################

# Make absolute cimports work.
#
# See
#     https://github.com/cython/cython/wiki/PackageHierarchy
#
# For example: my_include_dirs = [np.get_include()]
#my_include_dirs = ["."]
my_include_dirs = [".", np.get_include()]


# Choose the base set of compiler and linker flags.
#
if build_type == 'optimized':
    my_extra_compile_args_math    = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args_math       = extra_link_args_math_optimized
    my_extra_link_args_nonmath    = extra_link_args_nonmath_optimized
    my_debug = False
    print( "build configuration selected: optimized" )
elif build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args_math       = extra_link_args_math_debug
    my_extra_link_args_nonmath    = extra_link_args_nonmath_debug
    my_debug = True
    print( "build configuration selected: debug" )
else:
    raise ValueError("Unknown build configuration '%s'; valid: 'optimized', 'debug'" % (build_type))


def declare_cython_extension(extName, use_math=False, use_openmp=False, include_dirs=None):
    """Declare a Cython extension module for setuptools.

Parameters:
    extName : str
        Absolute module name, e.g. use `mylibrary.mypackage.mymodule`
        for the Cython source file `mylibrary/mypackage/mymodule.pyx`.

    use_math : bool
        If True, set math flags and link with ``libm``.

    use_openmp : bool
        If True, compile and link with OpenMP.

Return value:
    Extension object
        that can be passed to ``setuptools.setup``.
"""
    extPath = extName.replace(".", os.path.sep)+".pyx"

    # Filippo
    print('The path of {} is {}'.format(extName, extPath))

    if use_math:
        compile_args = list(my_extra_compile_args_math) # copy
        link_args    = list(my_extra_link_args_math)
        libraries    = ["m"]  # link libm; this is a list of library names without the "lib" prefix
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args    = list(my_extra_link_args_nonmath)
        libraries    = None  # value if no libraries, see setuptools.extension._Extension

    # OpenMP
    if use_openmp:
        #compile_args.insert( 0, *openmp_compile_args )
        compile_args[0:0] = openmp_compile_args
        link_args[0:0] = openmp_link_args

    # See
    #    http://docs.cython.org/src/tutorial/external.html
    #
    # on linking libraries to your Cython extensions.
    #
    return Extension( extName,
                      [extPath],
                      extra_compile_args=compile_args,
                      extra_link_args=link_args,
                      include_dirs=include_dirs,
                      libraries=libraries
                    )

# Gather user-defined data files
#
# http://stackoverflow.com/questions/13628979/setuptools-how-to-make-package-contain-extra-data-folder-and-all-folders-inside
datafiles = []
getext = lambda filename: os.path.splitext(filename)[1]
for datadir in datadirs:
    datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                       for root, dirs, files in os.walk(datadir)] )

# Add standard documentation (README et al.), if any, to data files
detected_docs = []
for docname in standard_docs:
    for ext in standard_doc_exts:
        filename = "".join( (docname, ext) )  # relative to the directory in which setup.py resides
        if os.path.isfile(filename):
            detected_docs.append(filename)
datafiles.append( ('.', detected_docs) )

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
import ast
init_py_path = os.path.join(libname, '__init__.py')
version = '0.0.unknown'
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            print( "WARNING: Version information not found in '%s', using placeholder '%s'" % (init_py_path, version), file=sys.stderr )
except MyFileNotFoundError:
    print( "WARNING: Could not find file '%s', using placeholder version information '%s'" % (init_py_path, version), file=sys.stderr )

#########################################################
# Set up modules
#########################################################

# declare Cython extension modules here
ext_module_awe2d = declare_cython_extension("serendipyty.seismic.modelling.awe2d", use_math=True, use_openmp=True, include_dirs=my_include_dirs)
ext_module_ebc = declare_cython_extension("serendipyty.seismic.modelling.ebc_filippo", use_math=True,  use_openmp=True, include_dirs=my_include_dirs)
ext_module_pml = declare_cython_extension("serendipyty.seismic.modelling.generate_pml_coeff", use_math=False, use_openmp=False, include_dirs=my_include_dirs)

# this is mainly to allow a manual logical ordering of the declared modules
#
#cython_ext_modules = [ext_module_pml]
cython_ext_modules = [ext_module_pml,
                      ext_module_awe2d,
                      ext_module_ebc]

# Call cythonize() explicitly, as recommended in the Cython documentation. See
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils
#
# This will favor Cython's own handling of '.pyx' sources over that provided by setuptools.
#
# Note that my_ext_modules is just a list of Extension objects. We could add any C sources (not coming from Cython modules) here if needed.
# cythonize() just performs the Cython-level processing, and returns a list of Extension objects.
#
my_ext_modules = cythonize(cython_ext_modules, include_path=my_include_dirs, gdb_debug=my_debug,
        compiler_directives={'embedsignature': True})

aa = 0
print('####### datafiles content is')
print(datafiles)
print('#######')
print(dir(my_ext_modules[aa]))
print('####### THE NAME IS {}'.format(cython_ext_modules[aa].name))
print('####### THE SOURCES ARE {}'.format(cython_ext_modules[aa].sources))
print('####### THE BLA BLA IS {}'.format(cython_ext_modules[aa].include_dirs))
print('####### THE BLA BLA IS {}'.format(cython_ext_modules[aa].library_dirs))
print('####### THE BLA BLA IS {}'.format(cython_ext_modules[aa].extra_compile_args))
print('####### THE BLA BLA IS {}'.format(cython_ext_modules[aa].extra_link_args))

#'extra_compile_args', 'extra_link_args', 'extra_objects', 'include_dirs', 'language', 'libraries', 'library_dirs', 'name', 'np_pythran', 'optional', 'py_limited_api', 'runtime_library_dirs', 'sources', 'swig_opts', 'undef_macros']

#########################################################
# Call setup()
#########################################################

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:
	# https://packaging.python.org/specifications/core-metadata/#name
    name = "serendipyty",

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
	# https://packaging.python.org/en/latest/single_source_version.html
    version = version,

    author = "Filippo Broggini",

    author_email = "filippo.broggini@erdw.ethz.ch",

    url = "https://github.com/serendipyty/serendipyty",

    description = SHORTDESC,
    long_description = DESC,

    # CHANGE THIS
    license = "Apache License 2.0",

    # free-form text field; http://stackoverflow.com/questions/34994130/what-platforms-argument-to-setup-in-setup-py-does
    platforms = ["Linux"],

    # See
    #    https://pypi.python.org/pypi?%3Aaction=list_classifiers
    #
    # for the standard classifiers.
    #
    # Remember to configure these appropriately for your project, especially license!
    #
    classifiers = [ "Development Status :: 1 - Planning",
                    "Environment :: Console",
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: Apache Software License",
                    "Operating System :: POSIX :: Linux",
                    "Programming Language :: Cython",
                    "Programming Language :: Python",
                    "Programming Language :: Python :: 3.4",
                    "Topic :: Scientific/Engineering",
                    "Topic :: Scientific/Engineering :: Mathematics",
                    "Topic :: Software Development :: Libraries",
                    "Topic :: Software Development :: Libraries :: Python Modules"
                  ],

    # See
    #    http://setuptools.readthedocs.io/en/latest/setuptools.html
    #
    setup_requires = ["cython", "numpy"],

    install_requires = ["numpy", "sphinx"],

    python_requires = '>=3.4',

    # keywords for PyPI (in case you upload your project)
    #
    # e.g. the keywords your project uses as topics on GitHub, minus "python" (if there)
    #
    keywords = ["setuptools template example cython"],

    # All extension modules (list of Extension objects)
    #
    ext_modules = my_ext_modules,

    # To automate compilation of cython files
    #
    cmdclass={'build_ext': build_ext},

    # Declare packages so that  python -m setup build  will copy .py files (especially __init__.py).
    #
    # This **does not** automatically recurse into subpackages, so they must also be declared.
    #
    packages = find_packages(exclude=['contrib', 'docs', 'tests*']),

    # Install also Cython headers so that other Cython modules can cimport ours
    #
    # Fileglobs relative to each package, **does not** automatically recurse into subpackages.
    #
    # FIXME: force sdist, but sdist only, to keep the .pyx files (this puts them also in the bdist)
    package_data = {'serendipyty': ['*.pxd', '*.pyx'],
                    'serendipyty.seismic': ['*.pxd', '*.pyx'],
                    'serendipyty.seismic.modelling': ['*.pxd', '*.pyx']},

    # Disable zip_safe, because:
    #   - Cython won't find .pxd files inside installed .egg, hard to compile libs depending on this one
    #   - dynamic loader may need to have the library unzipped to a temporary directory anyway (at import time)
    #
    zip_safe = False,

    # Custom data files not inside a Python package
    data_files = datafiles
)

