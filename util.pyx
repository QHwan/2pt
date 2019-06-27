import numpy as np

cimport cython
#cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double get_vac(double[:, :] vel0, double[:, :] vel1, double[:] m_vec, int n_row, int n_col):
    cdef int i, j
    cdef double sum = 0

    for i in range(n_row):
        for j in range(n_col):
            sum += m_vec[j] * vel0[i, j] * vel1[i, j]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:] center_of_mass(double[:,:] r_mat, double[:] m_vec):
    cdef int i, j, n_atoms
    cdef double m_tot = 0.
    cdef double[:] com_vec

    n_atoms = len(m_vec)
    com_vec = np.zeros(3)

    for i in range(n_atoms):
        m_tot += m_vec[i]

    for i in range(n_atoms):
        for j in range(3):
            com_vec[j] += m_vec[i]*r_mat[i,j]
    
    for i in range(3):
        com_vec[i] /= m_tot

    return com_vec