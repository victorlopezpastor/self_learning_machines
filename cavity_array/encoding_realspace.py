import numpy as np
from numpy.fft import fft2,ifft2


def encode(Psi,Theta):
    
    Phi = np.concatenate( [Psi, Theta], axis=0 )
    
    return Phi



def decode(Phi,N_psi=1):
    
    ans = Phi
    
    return [ ans[0:N_psi,:4,...], ans[N_psi:,...] ]