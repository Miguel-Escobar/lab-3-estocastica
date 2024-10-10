from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("p1_MH.pyx"),  # Reference to the Cython file
    include_dirs=[numpy.get_include()],  # Include NumPy headers
)