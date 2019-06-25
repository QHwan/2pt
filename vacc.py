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

def main():
    u = md.Universe('../lj/md.tpr', '../lj/md.trr')
    mols = u.select_atoms("all")

    vac_vec = vac(u, 'all', tau=u.trajectory[-1].time/2)

    np.savetxt('test.xvg', vac_vec, fmt='%5f')


