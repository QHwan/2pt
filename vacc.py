import math
import numpy as np
import copy
from numpy.fft import rfft, hfft
from scipy import optimize
import MDAnalysis as md
from tqdm import tqdm
from sys import getsizeof
import time
#from numba import jit, generated_jit, vectorize, float64
import cython
import util


u = md.Universe('../data/bulk_2pt/2pt.tpr', '../data/bulk_2pt/2pt.trr')
mols = u.select_atoms("all")

n_frames = len(u.trajectory)
tau = 5.
dt = u.trajectory[1].time - u.trajectory[0].time
atoms = u.select_atoms("name OW or name HW1 or name HW2")
n_atoms = len(atoms)
n_h2os = len(u.select_atoms("name OW"))
m_vec = mols.masses[0:3]

len_vac = int(tau/dt)+2
vac_trn_vec = np.zeros(len_vac)
vac_rot_vec = np.zeros(len_vac)
t_vec = np.zeros(len_vac)
for i in range(len_vac):
    t_vec[i] += dt*i

print("Decompose velocities into translational and rotational parts...")

v_3mat = np.zeros((n_frames, n_atoms, 3))
v_trn_3mat = np.zeros((n_frames, n_atoms, 3))
v_rot_3mat = np.zeros((n_frames, n_atoms, 3))
for i, ts in tqdm(enumerate(u.trajectory), total=n_frames, desc="velocity decompose"):
#for i, ts in enumerate(u.trajectory):
    v_3mat[i] = ts.velocities

    for j in range(n_h2os):
        v_trn_3mat[i,3*j] = util.center_of_mass(v_3mat[i,3*j:3*j+3], m_vec)
        v_trn_3mat[i,3*j+1:3*j+3] = v_trn_3mat[i,3*j]

        #for k in range(3):
        v_rot_3mat[i,3*j:3*j+3] = v_3mat[i,3*j:3*j+3] - v_trn_3mat[i,3*j:3*j+3]

print("Memory usage : {} Mb".format(3*getsizeof(v_3mat)/1024/1024))

for i in tqdm(range(n_frames - len_vac), desc="VAC calculation"):
#for i in range(n_frame-len_vac):
    v0_trn = v_trn_3mat[i]
    v0_rot = v_rot_3mat[i]
    for j in range(len_vac):
        vac_trn_vec[j] += util.get_vac(v0_trn, v_trn_3mat[i+j], m_vec, n_atoms, 3)
        vac_rot_vec[j] += util.get_vac(v0_rot, v_rot_3mat[i+j], m_vec, n_atoms, 3)

#vac_trn_vec *= m_vec[0]
#vac_rot_vec *= m_vec[0]
vac_trn_vec /= (n_frames - len_vac)*100
vac_rot_vec /= (n_frames - len_vac)*100

omat = np.transpose([t_vec, vac_trn_vec, vac_rot_vec])

np.savetxt('test.xvg', omat, fmt='%5f')


