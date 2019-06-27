import numpy as np
import matplotlib.pyplot as plt

imat = np.loadtxt("dos.xvg")
i2mat = np.loadtxt("dos2.xvg")
plt.plot(imat[:,0], imat[:,1], 'r')
plt.plot(i2mat[:,0], i2mat[:,1], 'g')
#plt.plot(imat[:,0], imat[:,1] + imat[:,2])
plt.show()