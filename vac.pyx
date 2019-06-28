import cython
cimport cython
import numpy as np
import numpy.linalg
cimport numpy as np
import sys
import math
import time
import sys

from tqdm import tqdm

import MDAnalysis as md
import matplotlib.pyplot as plt

cpdef main():

	cdef int i, j, k, l, n_h2os, n_frames, n_per_mol
	cdef int frame

	
	cdef float m_o, m_h, m_c, t, dt, M
	cdef float com_x, com_y, com_z, v_vib_x, v_vib_y, v_vib_z
	cdef float rr_x, rr_y, rr_z, I_xx, I_yy, I_zz, I_xy, I_yx, I_yz, I_zy, I_xz, I_zx, Ip_x, Ip_y, Ip_z, Ip_tot
	cdef float w_x, w_y, w_z, w2_x, w2_y, w2_z


	start = time.time()

	u = md.Universe(sys.argv[1], sys.argv[2])

	h2o = u.select_atoms("name OW or name HW1 or name HW2")
	ofilename = "vac.xvg"
	n_frames = len(u.trajectory)
	n_h2os = len(u.select_atoms("name OW"))
	t = 5
	dt = u.trajectory[1].time - u.trajectory[0].time

	m_o = 15.99998
	m_h = 1.00001
	m_c = 12.011
	M = (m_o+m_h*2)
	n_per_mol = 3
	cdef float[:] m_arr = np.array([m_o, m_h, m_h], dtype=np.float32)

	# Always check!!
	# C1, H1, H2, H3, O1, H4


	cdef float[:,:,:,:] r_mat = np.zeros((n_frames,n_h2os,n_per_mol,3), dtype=np.float32)

	cdef float[:,:,:,:] v_mat = np.zeros((n_frames,n_h2os,n_per_mol,3), dtype=np.float32)
	cdef float[:,:,:] v_trn_mat = np.zeros((n_frames,n_h2os,3), dtype=np.float32)
	cdef float[:,:,:] r_arr = np.zeros((n_h2os,n_per_mol,3), dtype=np.float32)
	cdef float[:,:,:] v_arr = np.zeros((n_h2os,n_per_mol,3), dtype=np.float32)
	cdef float[:,:,:,:] v_rot_mat = np.zeros((n_frames,n_h2os,n_per_mol,3), dtype=np.float32)

	cdef float[:,:] r_com_arr = np.zeros((n_per_mol,3), dtype=np.float32)
	cdef float[:,:] v_com_arr = np.zeros((n_per_mol,3), dtype=np.float32)

	cdef float[:] w_arr = np.zeros(3, dtype=np.float32)

	cdef float[:] v_rot_arr = np.zeros(3, dtype=np.float32)

	cdef float[:] r_com = np.zeros(3, dtype=np.float32)
	cdef float[:] v_com = np.zeros(3, dtype=np.float32)


	cdef float[:,:] I_mat = np.zeros((3,3), dtype=np.float32)
	cdef float[:,:] I_inv_mat = np.zeros((3,3), dtype=np.float32)
	cdef float[:,:,:] Ip_mat = np.zeros((n_frames,n_h2os,3), dtype=np.float32)
	cdef float[:,:] R_mat = np.zeros((3,3), dtype=np.float32)
	cdef float[:,:] t_mat = np.zeros((3,3), dtype=np.float32)

	cdef float[:] L_com_arr = np.zeros(3, dtype=np.float32)

	# Manually setup

	## Filling r_matrix
	cdef float[:,:] buf_mat = np.zeros((n_h2os*n_per_mol, 3), dtype=np.float32)
	for i, ts in enumerate(u.trajectory):
		buf_mat = h2o.positions
		for j in range (n_h2os):
			for k in range (n_per_mol):
				for l in range (3):
					r_mat[i,j,k,l] = buf_mat[n_per_mol*j+k, l] * 0.1

		buf_mat = h2o.velocities
		for j in range (n_h2os):
			for k in range (n_per_mol):
				for l in range (3):
					v_mat[i,j,k,l] = buf_mat[n_per_mol*j+k, l] * 0.1

	for l in tqdm(range(n_frames)):
		#Calculate molecule by molecule
		r_arr = r_mat[l]
		v_arr = v_mat[l]
		for i in range (n_h2os):
			#print i

			# Calculate COM of molecule
			r_com = np.zeros(3, dtype=np.float32)
			for j in range(n_per_mol):
				for k in range(3):
					r_com[k] += m_arr[j]*r_arr[i,j,k]

			for j in range(3):
				r_com[j] /= M

			# Calculate v_COM of molecule
			v_com = np.zeros(3, dtype=np.float32)
			for j in range(n_per_mol):
				for k in range(3):
					v_com[k] += m_arr[j]*v_arr[i,j,k]

			for j in range(3):
				v_com[j] /= M

			v_trn_mat[l,i] = v_com


			# Calculate relative position and velocity vector of molecule with respect to the COM
			r_com_arr = np.zeros((n_per_mol,3), dtype=np.float32)
			v_com_arr = np.zeros((n_per_mol,3), dtype=np.float32)
			for j in range(n_per_mol):
				for k in range(3):
					r_com_arr[j,k] = r_arr[i,j,k] - r_com[k]
					v_com_arr[j,k] = v_arr[i,j,k] - v_com[k]

			# Calculate I 
			I_mat = np.zeros((3,3), dtype=np.float32)
			for j in range(n_per_mol):
				I_mat[0,0] += m_arr[j]*(r_com_arr[j,1]*r_com_arr[j,1]+r_com_arr[j,2]*r_com_arr[j,2])
				I_mat[1,0] += m_arr[j]*(-r_com_arr[j,1]*r_com_arr[j,0])
				I_mat[1,1] += m_arr[j]*(r_com_arr[j,0]*r_com_arr[j,0]+r_com_arr[j,2]*r_com_arr[j,2])
				I_mat[2,0] += m_arr[j]*(-r_com_arr[j,0]*r_com_arr[j,2])
				I_mat[2,1] += m_arr[j]*(-r_com_arr[j,1]*r_com_arr[j,2])
				I_mat[2,2] += m_arr[j]*(r_com_arr[j,0]*r_com_arr[j,0]+r_com_arr[j,1]*r_com_arr[j,1])
			I_mat[0,1] = I_mat[1,0]
			I_mat[0,2] = I_mat[2,0]
			I_mat[1,2] = I_mat[2,1]
			I_inv_mat = np.linalg.inv(I_mat)

			# Calculate ww : L = m(r x v) = Iw
			L_com_arr = np.zeros(3, dtype=np.float32)
			for j in range(n_per_mol):
				L_com_arr[0] += m_arr[j]*(r_com_arr[j,1]*v_com_arr[j,2] - r_com_arr[j,2]*v_com_arr[j,1])
				L_com_arr[1] += m_arr[j]*(r_com_arr[j,2]*v_com_arr[j,0] - r_com_arr[j,0]*v_com_arr[j,2])
				L_com_arr[2] += m_arr[j]*(r_com_arr[j,0]*v_com_arr[j,1] - r_com_arr[j,1]*v_com_arr[j,0])

			# Calculate v_rot
			w_arr = np.zeros(3, dtype=np.float32)
			v_rot_arr = np.zeros(3, dtype=np.float32)

			w_arr[0] = I_inv_mat[0,0]*L_com_arr[0] + I_inv_mat[0,1]*L_com_arr[1] + I_inv_mat[0,2]*L_com_arr[2]
			w_arr[1] = I_inv_mat[1,0]*L_com_arr[0] + I_inv_mat[1,1]*L_com_arr[1] + I_inv_mat[1,2]*L_com_arr[2]
			w_arr[2] = I_inv_mat[2,0]*L_com_arr[0] + I_inv_mat[2,1]*L_com_arr[1] + I_inv_mat[2,2]*L_com_arr[2]

			for j in range(n_per_mol):
				v_rot_mat[l,i,j,0] += w_arr[1]*r_com_arr[j,2]-w_arr[2]*r_com_arr[j,1]
				v_rot_mat[l,i,j,1] += w_arr[2]*r_com_arr[j,0]-w_arr[0]*r_com_arr[j,2]
				v_rot_mat[l,i,j,2] += w_arr[0]*r_com_arr[j,1]-w_arr[1]*r_com_arr[j,0]



	cdef float[:] mvacf_trn_arr = np.zeros(int(t/dt), dtype=np.float32)
	cdef float[:] mvacf_rot_arr = np.zeros(int(t/dt), dtype=np.float32)

	cdef float[:,:] vdot_trn_arr = np.zeros((len(mvacf_trn_arr),n_h2os), dtype=np.float32)
	cdef float[:,:] vdot_rot_arr = np.zeros((len(mvacf_trn_arr),n_h2os), dtype=np.float32)

	cdef int len_mvacf_trn_arr
	len_mvacf_trn_arr = len(mvacf_trn_arr)

	for i in tqdm(range(n_frames-len_mvacf_trn_arr)):
		for j in range(len_mvacf_trn_arr):
			for k in range(n_h2os):
				vdot_trn_arr[j,k] += M*(v_trn_mat[i,k,0]*v_trn_mat[i+j,k,0]+v_trn_mat[i,k,1]*v_trn_mat[i+j,k,1]+v_trn_mat[i,k,2]*v_trn_mat[i+j,k,2])

				for l in range(n_per_mol):
					vdot_rot_arr[j,k] += m_arr[l]*(v_rot_mat[i,k,l,0]*v_rot_mat[i+j,k,l,0]+v_rot_mat[i,k,l,1]*v_rot_mat[i+j,k,l,1]+v_rot_mat[i,k,l,2]*v_rot_mat[i+j,k,l,2])


	for j in range (len_mvacf_trn_arr):
		for k in range (n_h2os):

			mvacf_trn_arr[j] += vdot_trn_arr[j,k]/(n_frames-len(mvacf_trn_arr))
			mvacf_rot_arr[j] += vdot_rot_arr[j,k]/(n_frames-len(mvacf_trn_arr))

	oarr = []
	for i in range (len_mvacf_trn_arr):
		oarr.append([dt*i, mvacf_trn_arr[i],mvacf_rot_arr[i]])



	np.savetxt(ofilename, oarr, fmt='%10f')

	end = time.time() - start

	print 'total time is ' + str(end)


