import numpy as np
import matplotlib.pyplot as plt

imat = np.loadtxt("dos.xvg")
plt.plot(imat[:,0], imat[:,1])
plt.plot(imat[:,0], imat[:,2])
plt.plot(imat[:,0], imat[:,1] + imat[:,2])
plt.show()