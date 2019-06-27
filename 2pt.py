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
    m_vec = mols.masses

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

    vac_vec *= m_vec[0]
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
start = time.time()

u = md.Universe('md.tpr', 'md.trr')
mols = u.select_atoms("all")
N = len(mols)
m = mols.masses[0]
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
h = 6.6261*1e-34 # (Js)
T = 131.744
N_avo = 6.02*1e23
V = vol
rho = N/vol
delta = 2*s0/9./N * 33.35641*1e-12  
delta *= (math.pi*k*T*N_avo/m/1e-3)**0.5 
delta *= (N/(V*1e-27))**(1./3.) # /nm3 -> /m3 
delta *= (6/math.pi)**(2./3.)
def f(x):
    func = 2*(delta**-4.5)*(x**7.5) 
    func -= 6*(delta**-3)*(x**5)
    func -= (delta**-1.5)*(x**3.5)
    func += 6*(delta**-1.5)*(x**2.5)
    func += 2*x-2
    return func
f = optimize.brentq(f, 0., 1.)
print(delta, f)

for i, freq in enumerate(freq_vec):
    dos_gas_vec[i] = s0/(1 + (math.pi*s0*freq/6/f/N)**2)

dos_solid_vec = dos_vec - dos_gas_vec

o_mat = np.transpose([freq_vec, dos_vec, dos_solid_vec, dos_gas_vec])
np.savetxt('gas.xvg', o_mat, fmt='%5f')

y = (f**2.5)*(delta**1.5)
z = (1+y+y**2-y**3)/((1-y)**3)  

S_HS = 2.5 + \
        math.log(((2*math.pi*m*1e-3*k*T)/(N_avo*h**2))**1.5*((V*1e-27)/(f*N))*z) + \
        (y*(3*y-4))/((1-y)**2)

# Setting bhv
bhv_vec = np.zeros(len(freq_vec))
for i, freq in enumerate(freq_vec):
    #bhv = (6.626*1.88365*freq_vec[i])/(1.3806*t)
    if i == 0:
        continue
    bhv_vec[i] = ((0.03*1e12*freq)*h/(k*T))

# Calculate Entropy
W_solid_vec = np.zeros(len(freq_vec))
W_gas_vec = np.zeros(len(freq_vec))

for i, bhv in enumerate(bhv_vec):
    
    if bhv == 0:
        W_solid = 0

    else:
        W_solid_vec[i] = (bhv/(math.exp(bhv)-1))-math.log(1-math.exp(-bhv))
        W_gas_vec[i] = (1./3.)*S_HS

W_solid_vec = np.multiply(dos_solid_vec, W_solid_vec)
W_gas_vec = np.multiply(dos_gas_vec, W_gas_vec)

S_solid = np.trapz(W_solid_vec, freq_vec)
S_gas = np.trapz(W_gas_vec, freq_vec)
S_total = S_solid + S_gas

print('S_solid : {} (J/mol/K)'.format(S_solid/N))
print('S_gas : {} (J/mol/K)'.format(S_gas/N))
print('S_total : {} (J/mol/K)'.format((S_total)/N))

#main()
print("Total time is {}s".format(time.time()-start))
