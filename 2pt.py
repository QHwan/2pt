import numpy as np
import MDAnalysis as md
import time
from tqdm import tqdm

import vac
import fft
import thermo

begin = time.time()

T = 300.
u = md.Universe('../data/bulk_2pt/2pt.tpr', '../data/bulk_2pt/2pt.trr')

S_trn_vec = []
S_rot_vec = []
S_tot_vec = []
for i in tqdm(range(50)):
    sel = "resnum {}".format(i)
    t_vec, vac_trn_vec, vac_rot_vec = vac.vac(u, sel)
    freq_vec, dos_trn_vec, dos_rot_vec = fft.dos(t_vec, vac_trn_vec, vac_rot_vec, T)

    V = u.dimensions[0]*u.dimensions[1]*u.dimensions[2]*1e-30/512
    S_trn, S_rot, S_tot = thermo.entropy(u, sel, freq_vec, dos_trn_vec, dos_rot_vec, T, V)
    #print(S_trn, S_rot, S_tot)
    S_trn_vec.append(S_trn)
    S_rot_vec.append(S_rot)
    S_tot_vec.append(S_tot)

    np.savetxt("dos.xvg", np.transpose([freq_vec, dos_trn_vec, dos_rot_vec]), fmt='%5f')
    exit(1)

print(np.average(S_trn_vec), np.std(S_trn_vec))
print(np.average(S_rot_vec), np.std(S_rot_vec))
print(np.average(S_tot_vec), np.std(S_tot_vec))


print("Total time is {}s.".format(time.time() - begin))

