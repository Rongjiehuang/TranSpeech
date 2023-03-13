from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'data_utils_fast',
  ext_modules = cythonize("data_utils_fast.pyx"),
  include_dirs=[numpy.get_include()]
)