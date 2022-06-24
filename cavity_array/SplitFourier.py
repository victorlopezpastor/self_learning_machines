import numpy as np

import encoding_realspace as enc

import eqs_motion as eom




def SplitFourier_step(Phi,params,dt):
    
    # returns the update of Psi and Theta after a step of Split-Fourier numerical integration
    
    Phi_1 = eom.LinearEv(Phi,params,dt/2)
    
    Phi_2 = eom.NonLinearEv(Phi_1,params,dt)
    
    Phi_3 = eom.LinearEv(Phi_2,params,dt/2)

    return Phi_3




def evolve(Psi_in,Theta_in,params,dt,N_steps):
    
    # returns the numerical solution of the eqs of motion using RK4 
    
    
    Phi_in = enc.encode(Psi_in,Theta_in)    # encode the inputs in Phi

    
    shp_phi = np.shape(Phi_in) + (N_steps,)    # create an array to store the evolution of the fields in time
    
    Phi = np.zeros(shp_phi,dtype=complex)
    
    Phi[...,0] = np.copy(Phi_in)
    
    for step in range(1,N_steps):    # simulate the evolution by the split-step method  
        
        Phi[...,step] = SplitFourier_step(Phi[...,step-1],params,dt)
    
    
    [Psi,Theta] = enc.decode(Phi,np.shape(Psi_in)[0])   # decode the output
    
    return [Psi,Theta]





def output_SLM(Psi_in,Theta_in,params,dt,N_steps):

    # returns the numerical solution of the eqs of motion using RK4 



    Phi = enc.encode(Psi_in,Theta_in)   # encode the inputs in Phi

    
    for step in range(1,N_steps):    # simulate the evolution by the split-step method  
        
        Phi = SplitFourier_step(Phi,params,dt)

    
    [Psi,Theta] = enc.decode(Phi,np.shape(Psi_in)[0])   # decode the output
    
        
    return [Psi,Theta]
    



