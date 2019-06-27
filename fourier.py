import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from scipy import fft

ifilename = "vac.xvg"
ofilename = "dos.xvg"
imat = np.loadtxt(ifilename)

t = imat[:,0]           # time vector
dt = t[1]-t[0]
Fs = 1.0/len(t)
y_trn = np.transpose(imat)[1]
y_rot = np.transpose(imat)[2]

t_mirror = np.zeros(2*len(t)-1)
y_trn_mirror = np.zeros(2*len(y_trn)-1)
y_rot_mirror = np.zeros(2*len(y_rot)-1)

for i in range (len(t)):
	t_mirror[i] += -1*t[len(t)-1-i]
	t_mirror[len(t)+i-1] += t[i]
for i in range (len(y_trn)):
	y_trn_mirror[i] += y_trn[len(t)-1-i]
	y_rot_mirror[i] += y_rot[len(t)-1-i]
	if i!=0:
		y_trn_mirror[len(t)+i-1] += y_trn[i]
		y_rot_mirror[len(t)+i-1] += y_rot[i]


t = t_mirror
y_trn = y_trn_mirror
y_rot = y_rot_mirror


n = len(y_trn)                       # length of the signal
# ps to cm-1
k = (np.arange(n)*3.33565*1e-11)/(n*dt*1e-12) # f = 1/(N*t), 1Hz = 3.33565*1e-11 cm-1
T = n/Fs
frq = k # two sides frequency range
freq = frq[range(int(n/2))]           # one side frequency range


#Y = np.fft.fft(y)/(n)              # fft computing and normalization
y_trn = np.fft.fft(y_trn)
y_rot = np.fft.fft(y_rot)
#print y_tot
#y_tot = y_tot*(2/(1.3805*1e-23*300))*(1000./(6.02*1e23))*(1/33.35641)*(0.004)
y_trn *= (1/(300*8.314*4))
y_rot *= (1/(300*8.314*4))

y_trn = y_trn[range(int(n/2))]
y_rot = y_rot[range(int(n/2))]

y_trn = abs(np.real(y_trn))
y_rot = abs(np.real(y_rot))

oarr=[]
for i in range (len(y_trn)):
	oarr.append([freq[i], y_trn[i],y_rot[i]])
np.savetxt(ofilename, oarr, fmt='%5f')
	
