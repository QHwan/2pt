import numpy as np

cimport cython
#cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_vac(double[:, :] vel0, double[:, :] vel1, int n_row, int n_col):
    cdef int i, j
    cdef double sum = 0

    for i in range(n_row):
        for j in range(n_col):
            sum += vel0[i, j] * vel1[i, j]
    return sum