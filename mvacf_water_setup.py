import numpy as np
from distutils.core import setup
from Cython.Build import cythonize

setup(
	name = "mvacf_water",
	include_dirs = [np.get_include()],
	ext_modules=cythonize('mvacf_water.pyx')
	)
