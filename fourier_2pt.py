import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from scipy import fft

ifile = sys.argv[1]
ofile = sys.argv[2]
imat = np.loadtxt(ifile)

t = np.transpose(imat)[0]            # time vector
dt = t[1]-t[0]
Fs = 1.0/len(t)
y_tot = np.transpose(imat)[1]
y_trn = np.transpose(imat)[2]
y_rot = np.transpose(imat)[3]
y_vib = np.transpose(imat)[4]

t_mirror = np.zeros(2*len(t)-1)
y_tot_mirror = np.zeros(2*len(y_tot)-1)
y_trn_mirror = np.zeros(2*len(y_trn)-1)
y_rot_mirror = np.zeros(2*len(y_rot)-1)
y_vib_mirror = np.zeros(2*len(y_vib)-1)

for i in range (len(t)):
	t_mirror[i] += -1*t[len(t)-1-i]
	t_mirror[len(t)+i-1] += t[i]
for i in range (len(y_tot)):
	y_tot_mirror[i] += y_tot[len(t)-1-i]
	y_trn_mirror[i] += y_trn[len(t)-1-i]
	y_rot_mirror[i] += y_rot[len(t)-1-i]
	y_vib_mirror[i] += y_vib[len(t)-1-i]
	if i!=0:
		y_tot_mirror[len(t)+i-1] += y_tot[i]
		y_trn_mirror[len(t)+i-1] += y_trn[i]
		y_rot_mirror[len(t)+i-1] += y_rot[i]
		y_vib_mirror[len(t)+i-1] += y_vib[i]

#plt.plot(t_mirror, y_mirror, 'ro')
#plt.show()
#exit(1)

t = t_mirror
y_tot = y_tot_mirror
y_trn = y_trn_mirror
y_rot = y_rot_mirror
y_vib = y_vib_mirror


n = len(y_tot)                       # length of the signal
# ps to cm-1
k = (np.arange(n)*3.33565*1e-11)/(n*dt*1e-12) # f = 1/(N*t), 1Hz = 3.33565*1e-11 cm-1
T = n/Fs
frq = k # two sides frequency range
freq = frq[range(n/2)]           # one side frequency range


#Y = np.fft.fft(y)/(n)              # fft computing and normalization
y_tot = np.fft.fft(y_tot)
y_trn = np.fft.fft(y_trn)
y_rot = np.fft.fft(y_rot)
y_vib = np.fft.fft(y_vib)
#print y_tot
#y_tot = y_tot*(2/(1.3805*1e-23*300))*(1000./(6.02*1e23))*(1/33.35641)*(0.004)
y_tot *= (1/(300*8.314*4))
y_trn *= (1/(300*8.314*4))
y_rot *= (1/(300*8.314*4))
y_vib *= (1/(300*8.314*4))

y_tot = y_tot[range(n/2)]
y_trn = y_trn[range(n/2)]
y_rot = y_rot[range(n/2)]
y_vib = y_vib[range(n/2)]

y_tot = abs(np.real(y_tot))
y_trn = abs(np.real(y_trn))
y_rot = abs(np.real(y_rot))
y_vib = abs(np.real(y_vib))



#plt.show()

oarr=[]
for i in range (len(y_tot)):
	oarr.append([freq[i], y_tot[i], y_trn[i],y_rot[i], y_vib[i]])
np.savetxt(sys.argv[2], oarr, fmt='%5f')
	
