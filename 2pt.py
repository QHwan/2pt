import numpy as np
import MDAnalysis as md
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pickle

import vac
import fft
import thermo

class Data:
    def __init__(self):
        self.index_res = 0
        self.ti = 0
        self.tf = 0

        self.T = 0
        self.V = 0

        self.freq_vec = None
        self.dos_trn_vec = None
        self.dos_rot_vec = None

        self.S_trn = 0
        self.S_rot = 0
        self.S_tot = 0

begin = time.time()

T = 300.
u = md.Universe('../data/bulk_2pt/2pt.tpr', '../data/bulk_2pt/2pt.trr')
V = u.dimensions[0]*u.dimensions[1]*u.dimensions[2]*1e-30/512


out = []

for i in tqdm(range(10)):
    data = Data()

    data.index_res = i
    data.T = T
    data.V = V

    sel = "resnum {}".format(i)

    t_vec, vac_trn_vec, vac_rot_vec = vac.vac(data, u, sel)
    fft.dos(data, t_vec, vac_trn_vec, vac_rot_vec, T)
    thermo.entropy(data, u, sel)

    out.append(data)


for i, data in enumerate(out):
    if i == 0:
        freq_vec = data.freq_vec
        dos_trn_vec = data.dos_trn_vec
        dos_rot_vec = data.dos_rot_vec
    else:
        dos_trn_vec += data.dos_trn_vec
        dos_rot_vec += data.dos_rot_vec

dos_trn_vec /= len(out)
dos_rot_vec /= len(out)

plt.plot(freq_vec, dos_trn_vec)
plt.plot(freq_vec, dos_rot_vec)
plt.plot(freq_vec, dos_trn_vec + dos_rot_vec)
plt.xlim((0, 1000))
plt.show()





