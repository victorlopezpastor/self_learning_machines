import numpy as np
from scipy.fftpack import dst


def dst2(Phi):
    
    return dst( dst( Phi, type=1, norm='ortho', axis=0 ), type=1, norm='ortho', axis=1 )




def Kerr(Phi,g,dt):
    
    # returns a Kerr nonlinear self-interaction
    
    return np.exp(-1j*g*np.abs(Phi)**2*dt)*Phi


def local_detuning(Phi,freq,dt):
    
    # returns a Kerr nonlinear self-interaction
    
    return np.einsum('ijk,ijk...->ijk...',np.exp(-1j*freq*dt),Phi)



def tigh_binding_2d(Phi,J,dt):
   
    # returns the rhs of the equation of motion for a 1d tight binding chain with coupling givne by J
    
    Phi_k = dst2(Phi) 
    
    N_cav = np.shape(Phi_k)[0]
    
    k = np.pi/(N_cav+1)*np.arange(1,N_cav+1)
    
    [kx,ky] = np.meshgrid(k,k)
    
    U = np.exp( -1j*2*dt*J*(   np.cos(kx) + np.cos(ky) ) )
    
    ans = dst2( np.einsum('ij,ij...->ij...', U,Phi_k) )
    
    return ans

    
    
def LinearEv(Phi,params,dt):
    
    # returns the rhs of the Hamilton's equation of motion for a system of interacting nonlinear cavities
    
    [g,J,freq] = params
    
    return tigh_binding_2d(Phi,J,dt)
    
   
    
    
    
def NonLinearEv(Phi,params,dt):
    
    # returns the rhs of the Hamilton's equation of motion for a system of interacting nonlinear cavities
    
    [g,J,freq] = params
    
    Phi_1 = local_detuning(Phi,freq,dt)
    
    return Kerr(Phi_1,g,dt)
 



'''
def Hamiltonian(Phi,params):
    
    # returns the value of the classical Hamiltonian for a given configuration
    
    [g,J] = params
    
    H = 2*J*np.sum( np.real( np.conj(Phi) * np.roll( Phi, 1, axis = 0 ) ), axis = (0,1) )
    
    H = H + 2*J*np.sum( np.real( np.conj(Phi) * np.roll( Phi, 1, axis = 1 ) ), axis = (0,1) )
    
    print(np.shape(H))
    
    return np.real( H + g/2 * np.sum( np.abs(Phi)**4, axis=(0,1) ) ) '''


def Hamiltonian(Phi,params):
    
    # returns the value of the classical Hamiltonian for a given configuration
    
    [g,J,freq] = params
    
    Phi_roll_x = np.roll( Phi, 1, axis=0 )
    
    Phi_roll_x[0,...] = 0.0 
    
    H = 2*J*np.sum( np.real( np.conj(Phi) * Phi_roll_x ), axis = (0,1) )
    
    print(np.shape(Phi))
    
    print(np.shape(H))
    
    Phi_roll_y = np.roll( Phi, 1, axis=1 )
    
    Phi_roll_y[:,0,...] = 0.0 
    
    H = H + 2*J*np.sum( np.real( np.conj(Phi) * Phi_roll_y ), axis = (0,1) )
    
    H = H + np.real( np.sum( np.einsum( 'ijk,ijk...->ijk...',freq,np.abs(Phi)**2 ), axis=(0,1) ) )
    
    return np.real( H + g/2 * np.sum( np.abs(Phi)**4, axis=(0,1) ) )
    
    
