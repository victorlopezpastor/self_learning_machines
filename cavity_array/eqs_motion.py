import numpy as np
from scipy.fftpack import dst

import encoding_realspace as enc


def dst2(Phi):
    
    return dst( dst( Phi, type=1, norm='ortho', axis=0 ), type=1, norm='ortho', axis=1 )




def Kerr(Phi,g,dt):
    
    # returns a Kerr nonlinear self-interaction
    
    return np.exp(-1j*g*np.abs(Phi)**2*dt)*Phi



def tigh_binding_2d(Phi,J,dt):
   
    # returns the rhs of the equation of motion for a 1d tight binding chain with coupling givne by J
    
    Phi_k = dst2(Phi) 
    
    N_cav = np.shape(Phi_k)[0]
    
    k = np.pi/(N_cav+1)*np.arange(1,N_cav+1)
    
    [kx,ky] = np.meshgrid(k,k)
    
    U = np.exp( -1j*2*dt*J*( np.cos(kx) + np.cos(ky) ) )
    
    ans = dst2( np.einsum('ij,ij...->ij...', U,Phi_k) )
    
    return ans

    
    
def LinearEv(Phi,params,dt):
    
    # returns the rhs of the Hamilton's equation of motion for a system of interacting nonlinear cavities
    
    [g,J,kappa,omega] = params
    
    [Psi,Theta] = enc.decode(Phi)
    
    Theta_1 = tigh_binding_2d(Theta,J,dt)
    
    return enc.encode(Psi,Theta_1)
    
   
    
def PsiTheta_coupling(Phi,kappa,omega,dt):
    
    [Psi,Theta] = enc.decode(Phi)
    
    width = np.shape(Theta)[0]
    
    index_x = np.array([0,0,width-1,width-1])
    
    index_y = np.array([0,width-1,0,width-1])
    
    Theta_ports = np.copy(Theta[index_x,index_y,...]) 
    
    Psi_0 = np.einsum( 'ij...,j->ij...',Psi,np.exp(-1j*omega*dt/2) )
    
    Psi_1 = np.cos(kappa*dt)*Psi_0 - 1j*np.sin(kappa*dt)*Theta_ports
    
    Theta_ports_1 = np.cos(kappa*dt)*Theta_ports - 1j*np.sin(kappa*dt)*Psi_0
    
    Theta[index_x,index_y,...] = Theta_ports_1
                      
    Psi_2 = np.einsum( 'ij...,j->ij...',Psi_1,np.exp(-1j*omega*dt/2) )
    
    return enc.encode(Psi_2,Theta)
    
    
    
    
def NonLinearEv(Phi,params,dt):
    
    # returns the rhs of the Hamilton's equation of motion for a system of interacting nonlinear cavities
    
    [g,J,kappa,omega] = params
    
    Phi_1 = Kerr(Phi,g,dt/2)
   
    Phi_2 = PsiTheta_coupling(Phi_1,kappa,omega,dt)

    Phi_3 = Kerr(Phi_2,g,dt/2)

    return Phi_3




def Hamiltonian(Phi,params):
    
    # returns the value of the classical Hamiltonian for a given configuration
    
    [g,J,kappa,omega] = params
    
    [Psi,Theta] = enc.decode(Phi)
    
    Theta_roll_x = np.roll( Theta, 1, axis=0 )
    
    Theta_roll_x[0,...] = 0.0 
    
    H = 2*J*np.sum( np.real( np.conj(Theta) * Theta_roll_x ), axis = (0,1) )
    
    Theta_roll_y = np.roll( Theta, 1, axis=1 )
    
    Theta_roll_y[:,0,...] = 0.0 
    
    H = H + 2*J*np.sum( np.real( np.conj(Theta) * Theta_roll_y ), axis = (0,1) )
    
    H = H + np.real( g/2 * np.sum( np.abs(Phi)**4, axis=(0,1) ) )
    
    
    width = np.shape(Theta)[0]
    
    index_x = np.array([0,0,width-1,width-1])
    
    index_y = np.array([0,width-1,0,width-1])
    
    Theta_ports = np.copy(Theta[index_x,index_y,...]) 
    
    H = H + 2*kappa*np.sum(np.real(np.conj(Psi[0,...])*Theta_ports), axis=0)
    
    H = H + np.einsum( 'j...,j->...',np.abs(Psi[0,...])**2,omega)
    
    return H
    
    
