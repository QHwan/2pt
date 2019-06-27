import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "vac",
	include_dirs = [np.get_include()],
	ext_modules=cythonize('vac.pyx')
	)
