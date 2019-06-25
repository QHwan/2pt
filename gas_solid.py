import math
import numpy as np
from numpy.fft import rfft, hfft
from scipy import optimize
import MDAnalysis as md
#from tqdm import tqdm
from sys import getsizeof
import time
#from numba import jit, generated_jit, vectorize, float64
import cython
import util


u = md.Universe('../lj/md.tpr', '../lj/md.trr')
mols = u.select_atoms("all")
box_vec = u.trajectory[0].dimensions

imat = np.loadtxt('dos.xvg')
freq_vec = imat[:,0]
dos_vec = imat[:,1]
dos_solid_vec = np.zeros(len(dos_vec))
dos_gas_vec = np.zeros(len(dos_vec))

# get dos_gas_vec
k = 1.3806*1e-23 # (J/K)
T = 131.744 # K
N_avo = 6.02*1e23
m = mols.masses[0]
h = 6.6261*1e-34 # (Js)
N = len(mols)
V = box_vec[0]*box_vec[1]*box_vec[2]/1000 # (nm3)

s0 = dos_vec[0]

Delta = 2*s0/9./N * 3.335641*1e-11  
Delta *= (math.pi*k*T*N_avo/m/1e-3)**(1./2.)
Delta *= (N/(V*1e-27))**(1./3.) # /nm3 -> /m3 
Delta *= (6/math.pi)**(2./3.)
def f(x):
    func = 2*(Delta**-4.5)*(x**7.5) 
    func -= 6*(Delta**-3)*(x**5)
    func -= (Delta**-1.5)*(x**3.5)
    func += 6*(Delta**-1.5)*(x**2.5)
    func += 2*x-2
    return func
f = optimize.brentq(f, 0., 1.)
print(Delta, f)

for i, freq in enumerate(freq_vec):
    dos_gas_vec[i] = s0/(1 + (math.pi*s0*freq/6/f/N)**2)
dos_solid_vec = dos_vec - dos_gas_vec

o_mat = np.transpose([freq_vec, dos_vec, dos_solid_vec, dos_gas_vec])
np.savetxt('gas.xvg', o_mat, fmt='%5f')
    


start = time.time()
#main()
print("Total time is {}s".format(time.time()-start))
