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

def vac(u, selection, tau):
    num_frame = len(u.trajectory)
    dt = u.trajectory[1].time - u.trajectory[0].time
    mols = u.select_atoms(selection)
    num_mol = len(mols)
    mass_vec = mols.masses

    len_vac = int(tau/dt)+2
    vac_vec = np.zeros(len_vac)
    t_vec = np.zeros(len_vac)
    for i in range(len_vac):
        t_vec[i] += dt*i

    vel_mat = np.zeros((num_frame, num_mol, 3))
    for i, ts in enumerate(u.trajectory):
        vel_mat[i] = ts.velocities

    print("Memory usage : {} Mb".format(getsizeof(vel_mat)/1024/1024))

    #for i in tqdm(range(num_frame - len_vac), desc="Loop of VAC calculation"):
    for i in range(num_frame-len_vac):
        vel0 = vel_mat[i]
        for j in range(len_vac):
            #vac_vec[j] += get_vac(vel0, vel_mat[i+j])
            vac_vec[j] += util.get_vac(vel0, vel_mat[i+j], num_mol, 3)

    vac_vec *= mass_vec[0]
    vac_vec /= (num_frame - len_vac)*100

    return np.transpose([t_vec, vac_vec])

'''
def main():
    u = md.Universe('../lj/md.tpr', '../lj/md.trr')
    mols = u.select_atoms("all")

    vac_vec = vac(u, 'all', tau=u.trajectory[-1].time/2)

    np.savetxt('test.xvg', vac_vec, fmt='%5f')
'''

'''
def main():
    imat = np.loadtxt('test.xvg')

    fft_vec = np.absolute(hfft(imat[:,1]))
    #kT = 2.479*131.744/298
    kT = 91.566065
    fft_vec *= 2/kT * 5 # why 5 ?

    vel_c = 299792458.  # m/s
    x_vec = np.zeros(len(fft_vec))
    df = 1/(imat[-1,0]*1e-12*vel_c) / 100 / 2. # 2. for hfft
    for i in range(len(x_vec)):
        x_vec[i] = i*df

    omat = np.transpose([x_vec, fft_vec])
    np.savetxt('dos.xvg', omat, fmt='%5f')
'''

u = md.Universe('../lj/md.tpr', '../lj/md.trr')
mols = u.select_atoms("all")
n_mols = len(mols)
mass = mols.masses[0]
box_vec = u.trajectory[0].dimensions
vol = box_vec[0]*box_vec[1]*box_vec[2]/1000

imat = np.loadtxt('dos.xvg')
freq_vec = imat[:,0]
dos_vec = imat[:,1]
dos_solid_vec = np.zeros(len(dos_vec))
dos_gas_vec = np.zeros(len(dos_vec))

# get dos_gas_vec
s0 = dos_vec[0]

kT = 91.566065
kT = 2.479 * 131.744/298. * 1e3
k = 1.3806*1e-23 # (J/k)
T = 131.744
N_avo = 6.02*1e23
V = vol
rho = n_mols/vol
Delta = 2*s0/9./n_mols * 33.35641*1e-12  
Delta *= (math.pi*k*T*N_avo/mass/1e-3)**0.5 
Delta *= (n_mols/(V*1e-27))**(1./3.) # /nm3 -> /m3 
Delta *= (6/math.pi)**(2./3.)
def f(x):
    func = 2*(Delta**-4.5)*(x**7.5) 
    func -= 6*(Delta**-3)*(x**5)
    func -= (Delta**-1.5)*(x**3.5)
    func += 6*(Delta**-1.5)*(x**2.5)
    func += 2*x-2
    return func
f = optimize.newton(f, 0.5)
print(Delta, f)

for i, freq in enumerate(freq_vec):
    dos_gas_vec[i] = s0/(1 + (math.pi*s0*freq/6/f/n_mols)**2)

o_mat = np.transpose([freq_vec, dos_vec, dos_gas_vec])
np.savetxt('gas.xvg', o_mat, fmt='%5f')
    


start = time.time()
#main()
print("Total time is {}s".format(time.time()-start))
