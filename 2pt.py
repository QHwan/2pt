import numpy as np
from numpy.fft import rfft, hfft
import MDAnalysis as md
from tqdm import tqdm
from sys import getsizeof
import time
#from numba import jit, generated_jit, vectorize, float64
import cython
import util

'''
@jit([float64(float64[:,:], float64[:,:])],
      nopython=True,
      fastmath=True,
      nogil=False,
      cache=True,
      parallel=False)
def get_vac(vel0, vel1):
    return np.sum(np.multiply(vel0, vel1))
'''

def vac(u, selection, tau=2.0):
    num_frame = len(u.trajectory)
    dt = u.trajectory[1].time - u.trajectory[0].time
    mols = u.select_atoms(selection)
    num_mol = len(mols)
    mass_vec = mols.masses

    len_vac = int(tau/dt)
    vac_vec = np.zeros(len_vac)
    t_vec = np.zeros(len_vac)
    for i in range(len_vac):
        t_vec[i] += dt*i

    vel_mat = np.zeros((num_frame, num_mol, 3))
    for i, ts in enumerate(u.trajectory):
        vel_mat[i] = ts.velocities

    print("Memory usage : {} Mb".format(getsizeof(vel_mat)/1024/1024))

    for i in tqdm(range(num_frame - len_vac), desc="Loop of VAC calculation"):
        vel0 = vel_mat[i]
        for j in range(len_vac):
            #vac_vec[j] += get_vac(vel0, vel_mat[i+j])
            vac_vec[j] += util.get_vac(vel0, vel_mat[i+j], num_mol, 3)

    vac_vec *= mass_vec[0]
    vac_vec /= (num_frame - len_vac)*100

    return np.transpose([t_vec, vac_vec])

'''
def main():
    u = md.Universe("md.tpr", "md.trr")
    mols = u.select_atoms("all")

    vac_vec = vac(u, "all", tau=3.0)

    np.savetxt("test.xvg", vac_vec, fmt="%5f")
'''

def main():
    imat = np.loadtxt("test.xvg")
    fft_vec = np.absolute(hfft(imat[:,1]))

    vel_c = 299792458.  # m/s
    x_vec = np.zeros(len(fft_vec))
    df = 1/(imat[-1,0]*1e-12*vel_c) / 100 / 2. # 2. for hfft
    for i in range(len(x_vec)):
        x_vec[i] = i*df

    omat = np.transpose([x_vec, fft_vec])
    print(fft_vec)
    np.savetxt("dos.xvg", omat, fmt="%5f")

start = time.time()
main()
print("Total time is {}s".format(time.time()-start))