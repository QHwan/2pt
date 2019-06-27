import numpy as np
import sys
import math
import scipy.optimize
from optparse import OptionParser as OP

def parsecmd(): 
	parser=OP()


	#parser.add_option('--volume',dest='volume',nargs=1,type='float',help='confinement')
	parser.add_option('--file,',dest='file',nargs=1,type='string',help='input file+_1.xvg')
	#parser.add_option('--ofile',dest='ofile',nargs=1,type='string',help='(1) ofile.xvg')
	parser.add_option('--nm',dest='nm',nargs=1,type='int',help='number of water molecule')
	parser.add_option('--temperature',dest='temperature',nargs=1,type='float',help='temperature')
	parser.add_option('--inertia',dest='inertia',nargs=3,type='float',help='principal moments of inertia')
	parser.add_option('--mass',dest='mass',nargs=1,type='float',help='mass of molecule')
	#parser.add_option('--energy',dest='energy',nargs=1,type='float',help='potential energy')


	(options, args) = parser.parse_args(sys.argv[1:])

	return options, args

options, args = parsecmd()



k = 1.3806*1e-23 # (J/ K)
T = options.temperature
N_avo = 6.02*1e23

m = options.mass # (g/mol)
h = 6.6261*1e-34 # (Js)

N = options.nm


I_px, I_py, I_pz = options.inertia

V_arr = [73.154,73.154,73.154,73.154]
E_arr = [0]

o_arr = []
S_tot_arr = []
S_trn_arr = []
S_rot_arr = []
S_vib_arr = []
for i in range (len(V_arr)):
	pt_filename = options.file

	pt_mat = np.loadtxt(pt_filename+'_'+str(i+1)+'.xvg')
	#rot_mat = np.loadtxt(rot_filename)
	v_arr = np.transpose(pt_mat)[0]
	s_trn_arr = np.transpose(pt_mat)[2]
	s_rot_arr = np.transpose(pt_mat)[3]
	s_vib_arr = np.transpose(pt_mat)[4]


	s_strn_arr = np.zeros(len(s_trn_arr))
	s_gtrn_arr = np.zeros(len(s_trn_arr))
	s_srot_arr = np.zeros(len(s_rot_arr))
	s_grot_arr = np.zeros(len(s_rot_arr))
	s_svib_arr = np.zeros(len(s_vib_arr))

	V = V_arr[i]
	E = E_arr[i]







	# Decompose s_arr -> s_s_arr and s_g_arr
	s0_trn = s_trn_arr[0]
	s0_rot = s_rot_arr[0]

	delta_trn = (s0_trn*(33.35641*1e-12))*((2)/(9.*N)) * \
			((math.pi*k*T*N_avo)/(m*1e-3))**(1./2) * \
			(N/(V*1e-27))**(1./3.) * \
			(6/math.pi)**(2./3.)
	delta_rot = (s0_rot*(33.35641*1e-12))*((2)/(9.*N)) * \
			((math.pi*k*T*N_avo)/(m*1e-3))**(1./2) * \
			(N/(V*1e-27))**(1./3.) * \
			(6/math.pi)**(2./3.)

	def func_trn(f):
		y = 2*(delta_trn**-4.5)*(f**7.5) - 6*(delta_trn**-3)*(f**5) - (delta_trn**-1.5)*(f**3.5) + 6*(delta_trn**-1.5)*(f**2.5) + 2*f - 2
		return y
	def func_rot(f):
		y = 2*(delta_rot**-4.5)*(f**7.5) - 6*(delta_rot**-3)*(f**5) - (delta_rot**-1.5)*(f**3.5) + 6*(delta_rot**-1.5)*(f**2.5) + 2*f - 2
		return y

 
	f_trn = scipy.optimize.brentq(func_trn,0,1)
	f_rot = scipy.optimize.brentq(func_rot,0,1)
	#f_rot = scipy.optimize.brentq(func_rot,0,0.2)
	print f_trn, f_rot


	y_trn = (f_trn**2.5)*(delta_trn**1.5)
	z_trn = (1+y_trn+y_trn**2-y_trn**3)/((1-y_trn)**3)

	y_rot = (f_rot**2.5)*(delta_rot**1.5)
	z_rot = (1+y_rot+y_rot**2-y_rot**3)/((1-y_rot)**3)

	# Hard Sphere entropy S_HS/k
	S_HS_trn = 2.5 + \
			   math.log(((2*math.pi*m*1e-3*k*T)/(N_avo*h**2))**1.5*((V*1e-27)/(f_trn*N))*z_trn) + \
			   (y_trn*(3*y_trn-4))/((1-y_trn)**2)

	TA = ((h)**2)/(8*(math.pi**2)*k*I_px*1e-18*1e-3/N_avo)
	TB = ((h)**2)/(8*(math.pi**2)*k*I_py*1e-18*1e-3/N_avo)
	TC = ((h)**2)/(8*(math.pi**2)*k*I_pz*1e-18*1e-3/N_avo)
	S_HS_rot = math.log((((math.pi**0.5)*(math.exp(1)**1.5))/(3))*(((T**3)/(TA*TB*TC))**0.5))
 
			   

	for j in range (len(s_gtrn_arr)):
		s_gtrn = s0_trn/(1+((math.pi*s0_trn*v_arr[j])/(6*f_trn*N))**2)
		s_gtrn_arr[j] += s_gtrn
		s_strn_arr[j] += s_trn_arr[j] - s_gtrn
		s_grot = s0_rot/(1+((math.pi*s0_rot*v_arr[j])/(6*f_rot*N))**2)
		s_grot_arr[j] += s_grot
		s_srot_arr[j] += s_rot_arr[j] - s_grot
		s_svib_arr = s_vib_arr
	#for j in range (len(s_grot_arr)):
	#	s_grot = s0_rot/(1+((math.pi*s0_rot*v_arr[j])/(6*f_rot*N))**2)
	#	s_grot_arr[j] += s_grot
	#	s_srot_arr[j] += s_rot_arr[j] - s_grot

	# Setting bhv
	bhv_arr = np.zeros(len(v_arr))
	for j in range (len(bhv_arr)):
		#bhv = (6.626*1.88365*v_arr[i])/(1.3806*t)
		if j == 0:
			bhv = 0
		else:
			bhv = ((0.03*1e12*v_arr[j])*h/(k*T))
		bhv_arr[j] += bhv
	



	# Calculate Entropy
	Ws_strn_arr = np.zeros(len(v_arr))
	Ws_gtrn_arr = np.zeros(len(v_arr))
	Ws_srot_arr = np.zeros(len(v_arr))
	Ws_grot_arr = np.zeros(len(v_arr))
	Ws_svib_arr = np.zeros(len(v_arr))


	for j in range (len(Ws_strn_arr)):
		bhv = bhv_arr[j]
		
		if bhv == 0:
			Ws_strn = 0
			Ws_srot = 0
			Ws_svib = 0

		else:
			Ws_strn = (bhv/(math.exp(bhv)-1))-math.log(1-math.exp(-bhv))
			Ws_srot = (bhv/(math.exp(bhv)-1))-math.log(1-math.exp(-bhv))
			Ws_svib = (bhv/(math.exp(bhv)-1))-math.log(1-math.exp(-bhv))


		Ws_strn_arr[j] += Ws_strn
		Ws_srot_arr[j] += Ws_srot
		Ws_svib_arr[j] += Ws_svib
		Ws_gtrn_arr[j] += (1./3.)*S_HS_trn
		Ws_grot_arr[j] += (1./3.)*S_HS_rot




	sWs_strn_arr = np.zeros(len(v_arr))
	sWs_gtrn_arr = np.zeros(len(v_arr))
	sWs_srot_arr = np.zeros(len(v_arr))
	sWs_grot_arr = np.zeros(len(v_arr))
	sWs_svib_arr = np.zeros(len(v_arr))

	for j in range (len(sWs_strn_arr)):

		sWs_strn = s_strn_arr[j]*Ws_strn_arr[j]
		sWs_gtrn = s_gtrn_arr[j]*Ws_gtrn_arr[j]
		sWs_srot = s_srot_arr[j]*Ws_srot_arr[j]
		sWs_grot = s_grot_arr[j]*Ws_grot_arr[j]
		sWs_svib = s_svib_arr[j]*Ws_svib_arr[j]



		sWs_strn_arr[j] += sWs_strn
		sWs_gtrn_arr[j] += sWs_gtrn
		sWs_srot_arr[j] += sWs_srot
		sWs_grot_arr[j] += sWs_grot
		sWs_svib_arr[j] += sWs_svib




	S_trn = np.trapz(sWs_strn_arr, v_arr) + np.trapz(sWs_gtrn_arr, v_arr)
	S_trn *= k*N_avo/N
	S_rot = np.trapz(sWs_srot_arr, v_arr) + np.trapz(sWs_grot_arr, v_arr)
	S_rot *= k*N_avo/N
	S_vib = np.trapz(sWs_svib_arr, v_arr)
	S_vib *= k*N_avo/N

	S_trn_arr.append(S_trn)
	S_rot_arr.append(S_rot)
	S_vib_arr.append(S_vib)
	S_tot_arr.append(S_trn+S_rot)




	#o_arr.append([n_arr[i], A_trn, A_rot, A_trn+A_rot, E_trn, E_rot, E_trn+E_rot, T*S_trn/1000, T*S_rot/1000, T*(S_trn+S_rot)/1000])
	print 'S_trn : ' + str(np.average(S_trn_arr)) + '\t' + str(np.std(S_trn_arr))
	print 'S_rot : ' + str(np.average(S_rot_arr)) + '\t' + str(np.std(S_rot_arr))
	#print 'S_vib : ' + str(np.average(S_vib_arr)) + '\t' + str(np.std(S_vib_arr))
	print 'S_tot : ' + str(np.average(S_tot_arr)) + '\t' + str(np.std(S_tot_arr))
	#o_arr = []
	#for i in range (len(s_gtrn_arr)):
	#	o_arr.append([v_arr[i], s_trn_arr[i], s_strn_arr[i], s_gtrn_arr[i]])
#np.savetxt(sys.argv[2],o_arr,fmt='%f')

