import numpy as np
import sys
import matplotlib.pyplot as plt
import math

ifilename = "vac.xvg"
ofilename = "dos.xvg"
imat = np.loadtxt(ifilename)

t = imat[:,0]           # time vector
dt = t[1]-t[0]
y_trn = imat[:,1]
y_rot = imat[:,2]

T = 300.

n = len(y_trn)                       # length of the signal
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

oarr=[]
for i in range (len(y_trn)):
	oarr.append([freq[i], y_trn[i],y_rot[i]])
np.savetxt(ofilename, oarr, fmt='%5f')

plt.plot(freq[:500], y_trn[:500])
plt.plot(freq[:500], y_rot[:500])
plt.xlim((0, 1000))
plt.ylim((0, 12))
plt.show()
	
