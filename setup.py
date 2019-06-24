import numpy as np
from distutils.core import setup
from Cython.Build import cythonize


setup(
    name = "test",
    include_dirs = [np.get_include()],
    ext_modules = cythonize("util.pyx", language_level=3)
)
