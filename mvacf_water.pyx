import cython
cimport cython
import numpy as np
import numpy.linalg
np.get_include()
cimport numpy as np
import sys
import math
import time

from tqdm import tqdm

import MDAnalysis as md

def main():

	cdef unsigned int i, j, k, l, num_mol, num_frame, n_per_mol
	cdef int frame

	
	cdef double m_o, m_h, m_c, t, dt, M
	cdef double com_x, com_y, com_z, v_com_x, v_com_y, v_com_z, v_vib_x, v_vib_y, v_vib_z
	cdef double rr_x, rr_y, rr_z, I_xx, I_yy, I_zz, I_xy, I_yx, I_yz, I_zy, I_xz, I_zx, Ip_x, Ip_y, Ip_z, Ip_tot
	cdef double w_x, w_y, w_z, w2_x, w2_y, w2_z


	start = time.time()

	u = md.Universe("..\data\2pt.tpr", "..\data\2pt.trr")

	ofilename = "vac.xvg"
	num_frame = len(u.trajectory)
	num_mol = len(u.select_atoms("name OW"))
	t = 5
	dt = u.trajectory[1].time - u.trajectory[0].time

	m_o = 15.99998
	m_h = 1.00001
	m_c = 12.011
	M = (m_o+m_h*2)
	n_per_mol = 3
	cdef double[:] mass_arr = np.array([m_o, m_h, m_h])

	# Always check!!
	# C1, H1, H2, H3, O1, H4


	cdef double[:,:,:,:] r_mat = np.zeros((num_frame,num_mol,n_per_mol,3))

	cdef double[:,:,:,:] v_mat = np.zeros((num_frame,num_mol,n_per_mol,3))
	cdef double[:,:,:] v_trn_mat = np.zeros((num_frame,num_mol,3))
	cdef double[:,:,:,:] w_mat = np.zeros((num_frame,num_mol,n_per_mol,3))
	cdef double[:,:,:] r_arr = np.zeros((num_mol,n_per_mol,3))
	cdef double[:,:,:] v_arr = np.zeros((num_mol,n_per_mol,3))
	cdef double[:,:,:,:] v_rot_mat = np.zeros((num_frame,num_mol,n_per_mol,3))
	cdef double[:,:,:,:] v_vib_mat = np.zeros((num_frame,num_mol,n_per_mol,3))
	cdef double[:,:,:] evec_x_mat = np.zeros((num_frame,num_mol,3))
	cdef double[:,:,:] evec_y_mat = np.zeros((num_frame,num_mol,3))
	cdef double[:,:,:] evec_z_mat = np.zeros((num_frame,num_mol,3))


	#cdef double[:,:,:] r_com_mat = np.zeros((num_frame,n_per_mol,3))
	cdef double[:,:] r_com_arr = np.zeros((n_per_mol,3))
	cdef double[:,:] v_com_arr = np.zeros((n_per_mol,3))
	cdef double[:,:] pr_com_arr = np.zeros((n_per_mol,3))

	cdef double[:] w_arr = np.zeros(3)

	cdef double[:] v_rot_arr = np.zeros(3)

	cdef double[:] v_com = np.zeros(3)


	cdef double[:,:] I_mat = np.zeros((3,3))
	cdef double[:,:] I_inv_mat = np.zeros((3,3))
	cdef double[:,:] evec = np.zeros((3,3))
	cdef double[:] Ip_arr = np.zeros(3)
	cdef double[:,:,:] Ip_mat = np.zeros((num_frame,num_mol,3))
	cdef double[:,:] R_mat = np.zeros((3,3))
	cdef double[:,:] a1_mat = np.zeros((3,3))
	cdef double[:] a2_mat = np.zeros(3)
	cdef double[:,:] a3_mat = np.zeros((3,3))
	cdef double[:,:] t_mat = np.zeros((3,3))

	cdef double[:,:] L_arr = np.zeros((n_per_mol,3))
	cdef double[:,:,:] L_mat = np.zeros((num_frame,num_mol,3))
	cdef double[:] L_com_arr = np.zeros(3)

	cdef list r_str_arr 
	cdef list v_str_arr


	# Manually setup

	## Filling r_matrix
	cdef double[:,:] buf_mat = np.zeros((num_mol*n_per_mol, 3))
	for i, ts in enumerate(u.trajectory):
		buf_mat = u.position
		for j in range (num_mol):
			for k in range (n_per_mol):
				for l in range (3):
					r_mat[i,j,k,l] = buf_mat[n_per_mol*j+k, l]

		buf_mat = u.velocity
		for j in range (num_mol):
			for k in range (n_per_mol):
				for l in range (3):
					v_mat[i,j,k,l] = buf_mat[n_per_mol*j+k, l]



	for l in tqdm(range(num_frame)):
		#Calculate molecule by molecule
		r_arr = r_mat[l]
		v_arr = v_mat[l]
		for i in range (num_mol):
			#print i

			# Calculate COM of molecule
			com_x = 0; com_y = 0; com_z = 0
			for j in range (n_per_mol):
				com_x += mass_arr[j]*r_arr[i,j,0]
				com_y += mass_arr[j]*r_arr[i,j,1]
				com_z += mass_arr[j]*r_arr[i,j,2]
			com_x /= M
			com_y /= M
			com_z /= M


			# Calculate v_COM of molecule
			v_com_x = 0; v_com_y = 0; v_com_z = 0
			for j in range (n_per_mol):
				v_com_x += mass_arr[j]*v_arr[i,j,0]
				v_com_y += mass_arr[j]*v_arr[i,j,1]
				v_com_z += mass_arr[j]*v_arr[i,j,2]
			v_com_x /= M
			v_com_y /= M
			v_com_z /= M

			v_trn_mat[l,i,0]+=v_com_x
			v_trn_mat[l,i,1]+=v_com_y
			v_trn_mat[l,i,2]+=v_com_z


			# Calculate relative position and velocity vector of molecule with respect to the COM
			r_com_arr = np.zeros((n_per_mol,3))
			v_com_arr = np.zeros((n_per_mol,3))
			for j in range (n_per_mol):
				r_com_arr[j,0] += r_arr[i,j,0] - com_x
				r_com_arr[j,1] += r_arr[i,j,1] - com_y
				r_com_arr[j,2] += r_arr[i,j,2] - com_z
				v_com_arr[j,0] += v_arr[i,j,0] - v_com_x
				v_com_arr[j,1] += v_arr[i,j,1] - v_com_y
				v_com_arr[j,2] += v_arr[i,j,2] - v_com_z

			# change r_mat -> r_com_mat
			#for j in range (n_per_mol):
			#	r_mat[frame,i,j,0] = r_com_arr[j,0]
			#	r_mat[frame,i,j,1] = r_com_arr[j,1]
			#	r_mat[frame,i,j,2] = r_com_arr[j,2]

			# Calculate I 
			I_mat = np.zeros((3,3))
			for j in range (n_per_mol):
				I_mat[0,0] += mass_arr[j]*(r_com_arr[j,1]*r_com_arr[j,1]+r_com_arr[j,2]*r_com_arr[j,2])
				I_mat[1,0] += mass_arr[j]*(-r_com_arr[j,1]*r_com_arr[j,0])
				I_mat[1,1] += mass_arr[j]*(r_com_arr[j,0]*r_com_arr[j,0]+r_com_arr[j,2]*r_com_arr[j,2])
				I_mat[2,0] += mass_arr[j]*(-r_com_arr[j,0]*r_com_arr[j,2])
				I_mat[2,1] += mass_arr[j]*(-r_com_arr[j,1]*r_com_arr[j,2])
				I_mat[2,2] += mass_arr[j]*(r_com_arr[j,0]*r_com_arr[j,0]+r_com_arr[j,1]*r_com_arr[j,1])
			I_mat[0,1] += I_mat[1,0]
			I_mat[0,2] += I_mat[2,0]
			I_mat[1,2] += I_mat[2,1]
			I_inv_mat = np.linalg.inv(I_mat)

			#Ip_mat[frame,i,0] += Ip_arr[0]
			#Ip_mat[frame,i,1] += Ip_arr[1]
			#Ip_mat[frame,i,2] += Ip_arr[2]

			#evec_x_mat[frame,i,0] += evec[0,0]
			#evec_x_mat[frame,i,1] += evec[1,0]
			#evec_x_mat[frame,i,2] += evec[2,0]
			#evec_y_mat[frame,i,0] += evec[0,1]
			#evec_y_mat[frame,i,1] += evec[1,1]
			#evec_y_mat[frame,i,2] += evec[2,1]
			#evec_z_mat[frame,i,0] += evec[0,2]
			#evec_z_mat[frame,i,1] += evec[1,2]
			#evec_z_mat[frame,i,2] += evec[2,2]


			#print Ip_arr


			# Calculate ww : L = m(r x v) = Iw
			#w_mat = np.zeros((num_frame,num_mol,n_per_mol,3))
			L_com_arr = np.zeros(3)
			for j in range (n_per_mol):
				L_com_arr[0] += mass_arr[j]*(r_com_arr[j,1]*v_com_arr[j,2] - r_com_arr[j,2]*v_com_arr[j,1])
				L_com_arr[1] += mass_arr[j]*(r_com_arr[j,2]*v_com_arr[j,0] - r_com_arr[j,0]*v_com_arr[j,2])
				L_com_arr[2] += mass_arr[j]*(r_com_arr[j,0]*v_com_arr[j,1] - r_com_arr[j,1]*v_com_arr[j,0])


			# Calculate v_vib
			#for j in range (n_per_mol):
			#	v_vib_mat[frame,i,j,0] += v_mat[frame,i,j,0] - v_trn_mat[frame,i,0] - \
			#				(w_mat[frame,i,j,1]*r_com_arr[j,2] - w_mat[frame,i,j,2]*r_com_arr[j,1])
			#	v_vib_mat[frame,i,j,1] += v_mat[frame,i,j,1] - v_trn_mat[frame,i,1] - \
			#				(w_mat[frame,i,j,2]*r_com_arr[j,0] - w_mat[frame,i,j,0]*r_com_arr[j,2])
			#	v_vib_mat[frame,i,j,2] += v_mat[frame,i,j,2] - v_trn_mat[frame,i,2] - \
			#				(w_mat[frame,i,j,0]*r_com_arr[j,1] - w_mat[frame,i,j,1]*r_com_arr[j,0])




			# Calculate v_rot
			w_arr = np.zeros(3)
			v_rot_arr = np.zeros(3)

			w_arr[0] += I_inv_mat[0,0]*L_com_arr[0] + I_inv_mat[0,1]*L_com_arr[1] + I_inv_mat[0,2]*L_com_arr[2]
			w_arr[1] += I_inv_mat[1,0]*L_com_arr[0] + I_inv_mat[1,1]*L_com_arr[1] + I_inv_mat[1,2]*L_com_arr[2]
			w_arr[2] += I_inv_mat[2,0]*L_com_arr[0] + I_inv_mat[2,1]*L_com_arr[1] + I_inv_mat[2,2]*L_com_arr[2]

			#L_mat[frame,i,0] += L_com_arr[0]
			#L_mat[frame,i,1] += L_com_arr[1]
			#L_mat[frame,i,2] += L_com_arr[2]
			for j in range (n_per_mol):
				v_rot_mat[l,i,j,0] += w_arr[1]*r_com_arr[j,2]-w_arr[2]*r_com_arr[j,1]
				v_rot_mat[l,i,j,1] += w_arr[2]*r_com_arr[j,0]-w_arr[0]*r_com_arr[j,2]
				v_rot_mat[l,i,j,2] += w_arr[0]*r_com_arr[j,1]-w_arr[1]*r_com_arr[j,0]


			#if i == 1:
			#	print v_trn_mat[frame,i]
			#	print v_rot_mat[frame,i]









	cdef double[:] mvacf_tot_arr = np.zeros(int(t/dt))
	cdef double[:] mvacf_trn_arr = np.zeros(int(t/dt))
	cdef double[:] mvacf_rot_arr = np.zeros(int(t/dt))
	cdef double[:] mvacf_rot_x_arr = np.zeros(int(t/dt))
	cdef double[:] buf_mvacf_rot_x_arr = np.zeros(int(t/dt))
	cdef double[:] mvacf_rot_y_arr = np.zeros(int(t/dt))
	cdef double[:] buf_mvacf_rot_y_arr = np.zeros(int(t/dt))
	cdef double[:] mvacf_rot_z_arr = np.zeros(int(t/dt))
	cdef double[:] buf_mvacf_rot_z_arr = np.zeros(int(t/dt))
	cdef double[:] mvacf_vib_arr = np.zeros(int(t/dt))

	cdef double[:,:] vdot_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] vdot_trn_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] vdot_rot_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] vdot_rot_x_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] vdot_rot_y_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] vdot_rot_z_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] buf_vdot_rot_x_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] buf_vdot_rot_y_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] buf_vdot_rot_z_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:,:] vdot_vib_arr = np.zeros((len(mvacf_trn_arr),num_mol))
	cdef double[:] t_arr = np.zeros(len(mvacf_trn_arr))

	cdef double buf_mvacf
	buf_mvacf = 0

	cdef int len_mvacf_trn_arr
	len_mvacf_trn_arr = len(mvacf_trn_arr)

	for i in range (num_frame-len_mvacf_trn_arr):
		if i%100 == 0:
			print i

		for j in range (len_mvacf_trn_arr):

			for k in range (num_mol):
				
				vdot_trn_arr[j,k] += M*(v_trn_mat[i,k,0]*v_trn_mat[i+j,k,0]+v_trn_mat[i,k,1]*v_trn_mat[i+j,k,1]+v_trn_mat[i,k,2]*v_trn_mat[i+j,k,2])

				for l in range (n_per_mol):
					vdot_rot_arr[j,k] += mass_arr[l]*(v_rot_mat[i,k,l,0]*v_rot_mat[i+j,k,l,0]+v_rot_mat[i,k,l,1]*v_rot_mat[i+j,k,l,1]+v_rot_mat[i,k,l,2]*v_rot_mat[i+j,k,l,2])









	for j in range (len_mvacf_trn_arr):
		for k in range (num_mol):

			#mvacf_tot_arr[j] += vdot_arr[j,k]/(num_frame-len(mvacf_trn_arr))

			mvacf_trn_arr[j] += vdot_trn_arr[j,k]/(num_frame-len(mvacf_trn_arr))
			mvacf_rot_arr[j] += vdot_rot_arr[j,k]/(num_frame-len(mvacf_trn_arr))











	oarr = []
	for i in range (len_mvacf_trn_arr):
		oarr.append([dt*i, mvacf_trn_arr[i],mvacf_rot_arr[i]])



	np.savetxt(ofilename, oarr, fmt='%10f')


	end = time.time() - start

	print 'total time is ' + str(end)


