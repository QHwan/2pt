import numpy as np

cimport cython
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_vac(double[:, :] vel0, double[:, :] vel1, int n_row, int n_col):
    cdef int i, j
    cdef double[:, :] vel2
    vel2 = np.zeros((n_row, n_col))

    for i in range(n_row):
        for j in range(n_col):
            vel2[i, j] = vel0[i, j] * vel1[i, j]
    return np.sum(vel2)