import numpy as np
import matplotlib.pyplot as plt

imat = np.loadtxt("vac.xvg")
plt.plot(imat[:,0], imat[:,1])
plt.plot(imat[:,0], imat[:,2])
plt.show()