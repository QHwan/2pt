import numpy as np
import sys
import math
import scipy.optimize
import MDAnalysis as md
import matplotlib.pyplot as plt

u = md.Universe("C:\\Development\data\md.tpr", "C:\\Development\data\md.trr")

k = 1.380649*1e-23 # (J/ K)
T = 300.
N_avo = 6.02214*1e23

m = np.sum(u.select_atoms("name OW or name HW1 or name HW2").masses[0:3]) * 1e-3 / N_avo # (kg)
h = 6.62607*1e-34 # (Js)

N = len(u.select_atoms("name OW"))
V = u.dimensions[0]*u.dimensions[1]*u.dimensions[2] * 1e-30 # m3

I_px, I_py, I_pz = 0.5926, 1.3334, 1.926

V_arr = [V]
E_arr = [0]

o_arr = []
S_tot_arr = []
S_trn_arr = []
S_rot_arr = []
for i in range(len(V_arr)):
	pt_mat = np.loadtxt("dos.xvg")
	v_arr = pt_mat[:,0]
	s_trn_arr = pt_mat[:,1]
	s_rot_arr = pt_mat[:,2]

	V = V_arr[i]
	E = E_arr[i]

	v_arr = v_arr[:300]
	s_trn_arr = s_trn_arr[:300]
	s_rot_arr = s_rot_arr[:300]

	# Decompose s_arr -> s_s_arr and s_g_arr
	s0_trn = s_trn_arr[0]
	s0_rot = s_rot_arr[0]

	delta_trn = (s0_trn*3.33565*1e-11)*(2/9./N) * \
			(math.pi*k*T/m)**(1./2) * \
			(N/V)**(1./3.) * \
			(6/math.pi)**(2./3.)
	delta_rot = (s0_rot*3.35565*1e-11)*(2/9./N) * \
			(math.pi*k*T/m)**(1./2) * \
			(N/V)**(1./3.) * \
			(6/math.pi)**(2./3.)

	def func_trn(f):
		y = 2*(delta_trn**-4.5)*(f**7.5) - 6*(delta_trn**-3)*(f**5) - (delta_trn**-1.5)*(f**3.5) + 6*(delta_trn**-1.5)*(f**2.5) + 2*f - 2
		return y
	def func_rot(f):
		y = 2*(delta_rot**-4.5)*(f**7.5) - 6*(delta_rot**-3)*(f**5) - (delta_rot**-1.5)*(f**3.5) + 6*(delta_rot**-1.5)*(f**2.5) + 2*f - 2
		return y
 
	f_trn = scipy.optimize.brentq(func_trn,0,1)
	f_rot = scipy.optimize.brentq(func_rot,0,1)
	print("f_trn = {}, f_rot = {}".format(f_trn, f_rot))

	y_trn = (f_trn**2.5)/(delta_trn**1.5)
	z_trn = (1 + y_trn + y_trn**2 - y_trn**3)/((1 - y_trn)**3)

	y_rot = (f_rot**2.5)/(delta_rot**1.5)
	z_rot = (1 + y_rot + y_rot**2 - y_rot**3)/((1 - y_rot)**3)

	# Hard Sphere entropy S_HS/k
	S_HS_trn = 2.5 + math.log( (2*math.pi*m*k*T/h/h)**1.5  *V/f_trn/N*z_trn ) + y_trn*(3*y_trn-4)/(1-y_trn)**2

	TA = (h**2)/(8*(math.pi**2)*k*I_px*1e-18*1e-3/N_avo)
	TB = (h**2)/(8*(math.pi**2)*k*I_py*1e-18*1e-3/N_avo)
	TC = (h**2)/(8*(math.pi**2)*k*I_pz*1e-18*1e-3/N_avo)
	S_HS_rot = math.log((((math.pi**0.5)*(math.exp(1)**1.5))/(3))*(((T**3)/(TA*TB*TC))**0.5)) 
			
	s_trn_gas_arr = s0_trn / (1 + (math.pi*s0_trn*v_arr/6/f_trn/N)**2)
	s_trn_sol_arr = s_trn_arr - s_trn_gas_arr
	s_rot_gas_arr = s0_rot / (1 + (math.pi*s0_rot*v_arr/6/f_rot/N)**2)
	s_rot_sol_arr = s_rot_arr - s_rot_gas_arr 



	# Setting bhv
	bhv_arr = 2.9979*1e10*v_arr*h/k/T

	print(v_arr)

	# Calculate Entropy
	W_trn_sol_arr = np.zeros(len(v_arr))
	W_trn_gas_arr = np.zeros(len(v_arr))
	W_rot_sol_arr = np.zeros(len(v_arr))
	W_rot_gas_arr = np.zeros(len(v_arr))

	for j, bhv in enumerate(bhv_arr):
		if j != 0:
			W_trn_sol_arr[j] = bhv/(math.exp(bhv)-1) - math.log(1-math.exp(-bhv))
			W_rot_sol_arr[j] = bhv/(math.exp(bhv)-1) - math.log(1-math.exp(-bhv))
			W_trn_gas_arr[j] = (1./3.)*S_HS_trn
			W_rot_gas_arr[j] = (1./3.)*S_HS_rot

	'''
	plt.plot(v_arr, W_trn_sol_arr)
	plt.plot(v_arr, s_trn_sol_arr)
	plt.plot(v_arr, np.multiply(s_trn_sol_arr, W_trn_sol_arr))
	plt.plot(v_arr, W_trn_gas_arr)
	plt.plot(v_arr, s_trn_gas_arr)
	plt.plot(v_arr, np.multiply(s_trn_gas_arr, W_trn_gas_arr))
	plt.xlim((0, 1000))
	plt.ylim((0, 15))
	plt.show()
	'''

	S_trn_sol = np.trapz(np.multiply(s_trn_sol_arr, W_trn_sol_arr), v_arr) * k*N_avo/N
	S_trn_gas = np.trapz(np.multiply(s_trn_gas_arr, W_trn_gas_arr), v_arr) * k*N_avo/N
	S_trn = S_trn_sol + S_trn_gas
	print("Translational entropy: solid {}, gas {}, total {}".format(S_trn_sol, S_trn_gas, S_trn))
	print(np.trapz(np.multiply(s_trn_arr, W_trn_sol_arr), v_arr) * k*N_avo/N)

	S_rot_sol = np.trapz(np.multiply(s_rot_sol_arr, W_rot_sol_arr), v_arr) * k*N_avo/N
	S_rot_gas = np.trapz(np.multiply(s_rot_gas_arr, W_rot_gas_arr), v_arr) * k*N_avo/N
	S_rot = S_rot_sol + S_rot_gas
	print("Rotational entropy: solid {}, gas {}, total {}".format(S_rot_sol, S_rot_gas, S_rot))
	print(np.trapz(np.multiply(s_rot_arr, W_trn_sol_arr), v_arr) * k*N_avo/N)


	S_trn_arr.append(S_trn)
	S_rot_arr.append(S_rot)
	S_tot_arr.append(S_trn+S_rot)


