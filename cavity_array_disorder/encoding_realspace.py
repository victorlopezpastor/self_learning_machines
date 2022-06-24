import numpy as np
from numpy.fft import fft2,ifft2


def encode(Psi,Theta):
    
    Phi = np.concatenate( [Psi, Theta], axis=0 )
    
    return Phi



def decode(Phi,N_psi):
    
    ans = Phi
    
    return [ ans[0:N_psi,...], ans[N_psi:,...] ]