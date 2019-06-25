import math
import numpy as np
from numpy.fft import fft, rfft, hfft
from scipy import optimize
import MDAnalysis as md
#from tqdm import tqdm
from sys import getsizeof
import time
#from numba import jit, generated_jit, vectorize, float64
import cython
import util

def main():
    imat = np.loadtxt('test.xvg')

    t_vec = imat[:,0]
    vac_vec = imat[:,1]
    dt = t_vec[1] - t_vec[0]
    freq_s = 1.0/len(t_vec)

    freq_vec = np.zeros(2*len(t_vec)-1)
    fft_vec = np.zeros(2*len(t_vec)-1)
    for i in range(len(t_vec)):
        freq_vec[i] += -1*t_vec[len(t_vec)-1-i]
        freq_vec[len(t_vec)+i-1] += t_vec[i]
    for i in range(len(t_vec)):
        fft_vec[i] += vac_vec[len(t_vec)-1-i]
        if i != 0:
            fft_vec[len(t_vec)+i-1] += vac_vec[i]

    n = len(fft_vec)
    k = (np.arange(n)*3.33565*1e-11)/(n*dt*1e-12) # f = 1/(N*t), 1 Hz = 3.33565*1e-11 cm-1
    T = 131.744
    freq_vec = k # two sides frequency range
    freq_vec = freq_vec[range(n/2)]

    fft_vec = fft(fft_vec)
    fft_vec *= (1/(131*8.314*4))
    fft_vec = fft_vec[range(n/2)]
    fft_vec = abs(np.real(fft_vec))
    print(len(freq_vec), len(fft_vec))

    omat = np.transpose([freq_vec, fft_vec])
    np.savetxt('dos.xvg', omat, fmt='%5f')

main()
