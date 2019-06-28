import numpy as np
import sys
import matplotlib.pyplot as plt
import math

def dos_now_checking(t, y_trn, y_rot, T):
	dt = t[1]-t[0]

	n = len(t)                       # length of the signal
	# ps to cm-1
	freq = 0.5 * (np.arange(n)*3.33565*1e-11)/(n*dt*1e-12) # f = 1/(N*t), 1Hz = 3.33565*1e-11 cm-1
	#freq = frq[range(int(n/2))]           # one side frequency range


	# fft computing and normalization
	y_trn = np.fft.hfft(y_trn)
	y_rot = np.fft.hfft(y_rot)
	y_trn *= (1/(T*8.314*4))
	y_rot *= (1/(T*8.314*4))

	y_trn = y_trn[:freq.size]
	y_rot = y_rot[:freq.size]

	return freq, y_trn, y_rot

	'''
	plt.plot(freq[:500], y_trn[:500])
	plt.plot(freq[:500], y_rot[:500])
	plt.xlim((0, 1000))
	plt.ylim((0, 12))
	plt.show()
	'''
		

def dos(t, y_trn, y_rot, T):
	dt = t[1]-t[0]

	n = len(t)                       # length of the signal

	t_mirror = np.zeros(2*n-1)
	y_trn_mirror = np.zeros(2*n-1)
	y_rot_mirror = np.zeros(2*n-1)

	for i in range (n):
		t_mirror[i] += -1*t[n-1-i]
		t_mirror[n+i-1] += t[i]

	for i in range (n):
		y_trn_mirror[i] += y_trn[n-1-i]
		y_rot_mirror[i] += y_rot[n-1-i]
		if i!=0:
			y_trn_mirror[n+i-1] += y_trn[i]
			y_rot_mirror[n+i-1] += y_rot[i]

	t = t_mirror
	y_trn = y_trn_mirror
	y_rot = y_rot_mirror
	# ps to cm-1
	freq = (np.arange(n)*3.33565*1e-11)/(n*dt*1e-12) # f = 1/(N*t), 1Hz = 3.33565*1e-11 cm-1
	#freq = frq[range(int(n/2))]           # one side frequency range


	# fft computing and normalization
	y_trn = np.fft.fft(y_trn)
	y_rot = np.fft.fft(y_rot)
	y_trn *= (1/(T*8.314*4))
	y_rot *= (1/(T*8.314*4))

	freq = freq[(range(int(n/2)))]
	y_trn = y_trn[(range(int(n/2)))]
	y_rot = y_rot[(range(int(n/2)))]

	y_trn = np.abs(np.real(y_trn))
	y_rot = np.abs(np.real(y_rot))

	return freq, y_trn, y_rot

	'''
	plt.plot(freq[:500], y_trn[:500])
	plt.plot(freq[:500], y_rot[:500])
	plt.xlim((0, 1000))
	plt.ylim((0, 12))
	plt.show()
	'''