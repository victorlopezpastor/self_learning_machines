import numpy as np


def normalize(Psi):
    
    return Psi/np.sqrt( np.sum( np.abs(Psi)**2, axis=(0,1) ) + 1e-14 )
    




def XOR_fields(width,N_psi,M,amp_psi):
    
    # returns two random gaussian vectors with size D_psi and D_theta
    
    # Theta is initialized with zero momentum (purely real)
    
    
    Psi = np.zeros((N_psi,width,4),dtype=complex)
    
    Psi[0,:2,0] = [1.0,1.0]
    
    Psi[0,:2,1] = [1.0,-1.0]
    
    Psi[0,:2,2] = [-1.0,1.0]
    
    Psi[0,:2,3] = [-1.0,-1.0]

    Psi =  Psi[:,:,:,np.newaxis]*amp_psi[np.newaxis,np.newaxis,np.newaxis,:]
    
    
    Theta = np.random.random([width,width]) + 1j*0.0
    
    Theta = normalize( Theta )[:,:,np.newaxis]*(np.ones(M)[np.newaxis,np.newaxis,:])
    
    
    target = np.ones((4,M),dtype=complex)
    
    target[0,:] = 1.0
    
    target[1,:] = -1.0
    
    target[2,:] = -1.0
    
    target[3,:] = 1.0
    
    target = 0.5*target*amp_psi[np.newaxis,:]
    
    
    return [Psi,Theta,target]



