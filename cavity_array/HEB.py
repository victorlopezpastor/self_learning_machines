import numpy as np
from tqdm import tqdm

import SplitFourier as SF







def C(Psi_out,target):
    
    # computes the MSE cost function
    
    return np.abs(Psi_out[0,-1,...]-target)**2
    
  

    
def dC(Psi_out,target):
    
    # computes the complex derivative of the MSE cost function with respect to the output 
    # (i.e. the perturbation in the output)

    ans = 0.0*Psi_out
    
    ans[0,-1] = ( Psi_out[0,-1,...] - target )
    
    return ans






def TR(Psi):
    
    return np.conj( Psi )
    
    

def echo_step(Psi_in,Theta_in,params,target,eta,dt,N_steps):
    
    # implements the Hamiltonian part of HEB: forward pass + backward pass
    
    # returns the output of the echo step and the cost function
 
    
    
    # forward step 
    
    [Psi_fw_out,Theta_fw_out] = SF.output_SLM(Psi_in,Theta_in,params,dt,N_steps)
    
    
    # perturbation of the output and phase conjugation operation
    
    Psi_bw_in = TR( Psi_fw_out - 1j*eta*dC(Psi_fw_out,target) )
    
    Theta_bw_in = TR(Theta_fw_out)
    
    cost = C(Psi_fw_out,target)
    
    
    # backward step 

    [Psi_bw_out,Theta_bw_out] = SF.output_SLM(Psi_bw_in,Theta_bw_in,params,dt,N_steps)
    
    
    return [TR(Psi_bw_out),TR(Theta_bw_out),cost]
    
    
def echo_step_rec(Psi_in,Theta_in,params,target,eta,dt,N_steps):
    
    # implements the Hamiltonian part of HEB: forward pass + backward pass
    
    # returns the output of the echo step and the cost function
 
    
    
    # forward step 
    
    [Psi_fw_out,Theta_fw_out] = SF.output_SLM(Psi_in,Theta_in,params,dt,N_steps)
    
    
    # perturbation of the output and phase conjugation operation
    
    Psi_bw_in = TR( Psi_fw_out - 1j*eta*dC(Psi_fw_out,target) )
    
    Theta_bw_in = TR(Theta_fw_out)
    
    cost = C(Psi_fw_out,target)
    
    
    # backward step 

    [Psi_bw_out,Theta_bw_out] = SF.output_SLM(Psi_bw_in,Theta_bw_in,params,dt,N_steps)
    
    
    return [TR(Psi_bw_out),TR(Theta_bw_out),cost,Psi_fw_out]
    
    
    
    
def decay_step(Theta_in,loss,m):
    
    # implements the decay step: in this case we use a simple friction-like dissipation

    # loss  is the damping coefficient x timestep:    p_f = p_i exp(-loss)
    
    kick = 1.0/m*np.imag(Theta_in)*( 1 - np.exp(-loss) )
    
    # cutoff 
    
    cutoff = 1.0e-1
    
    kick = kick/np.sqrt( 1.0 + np.abs(kick)**2/cutoff**2 )
    
    theta = np.real(Theta_in) + kick
    
    pi_theta = np.imag(Theta_in)*np.exp(-loss) 
    
    return theta + 1j*pi_theta


    
    
    
def HEB_step(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m):
    
    # implements a full HEB step: echo step + decay step
    # Psi_in, Theta_in   are the   initial conditions
    
    
    [Psi,Theta,cost] = echo_step(Psi_in,Theta_in,params,target,eta,dt,N_steps)
    
    Theta = decay_step(Theta,loss,m)
    
    return [Psi,Theta,cost];



def HEB_step_av(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m):
    
    # implements a full HEB step: echo step + decay step
    # Psi_in, Theta_in   are the   initial conditions
    
    
    [Psi,Theta,cost] = echo_step(Psi_in,Theta_in,params,target,eta,dt,N_steps)
    
    Theta = decay_step(Theta,loss,m)
    
    [Psi,Theta,cost] = echo_step(Psi_in,Theta,params,target,-eta,dt,N_steps)
    
    Theta = decay_step(Theta,loss,-m)
    
    return [Psi,Theta,cost];


def HEB_step_av_rec(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m):
    
    # implements a full HEB step: echo step + decay step
    # Psi_in, Theta_in   are the   initial conditions
    
    
    [Psi,Theta,cost,Psi_out] = echo_step_rec(Psi_in,Theta_in,params,target,eta,dt,N_steps)
    
    output = np.copy(Psi_out[0,-1,...])
    
    Theta = decay_step(Theta,loss,m)
    
    [Psi,Theta,cost] = echo_step(Psi_in,Theta,params,target,-eta,dt,N_steps)
    
    Theta = decay_step(Theta,loss,-m)
    
    return [Psi,Theta,cost,output];



def HEB_step_mom(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m):
    
    # implements a full HEB step: echo step + decay step
    # Psi_in, Theta_in   are the   initial conditions
    
    
    [Psi,Theta,cost] = echo_step(Psi_in,Theta_in,params,target,eta,dt,N_steps)
    
    Theta = np.conj( decay_step(Theta,loss,m) )
    
    [Psi,Theta,cost] = echo_step(Psi_in,Theta,params,target,-eta,dt,N_steps)
    
    Theta = np.conj( decay_step(Theta,loss,1e10) )
    
    return [Psi,Theta,cost];




def grad_C_fd(Psi_in,Theta_in,params,dt,N_steps,target,eta,x,y,h):
    
    # computes the gradient of the cost function using the finite difference method
    
    # Psi_in and Theta_in are fields 1D
    
    
    Theta_in_b  = np.copy(Theta_in)
    
    Theta_in_b[x,y,...] = Theta_in[x,y,...] + h 
    
    [Psi_fw,Theta_fw] = SF.output_SLM(Psi_in,Theta_in_b,params,dt,N_steps)
    
    C_b = C(Psi_fw,target)
    

    
    Theta_in_a  = np.copy(Theta_in)
    
    Theta_in_a[x,y,...] = Theta_in[x,y,...] - h 
    
    [Psi_fw,Theta_fw] = SF.output_SLM(Psi_in,Theta_in_a,params,dt,N_steps)
    
    C_a = C(Psi_fw,target)
    
    
    return [ 1.0/(2*h)*( C_b - C_a ), 0.5*( C_b + C_a ) ] 





def optimization(Psi_in_dset,Theta_in,params,dt,N_steps,target_dset,eta,loss,m,N_train):
    
    # optimization loop 
    # implements N_train iterations of HEB
     
    [N_samples,N_traj] = np.shape(Psi_in_dset)[-2:] 
    
    cost = np.zeros((N_train,N_traj))
    
    for step in tqdm(range(N_train)):
        
        sample_index = np.random.randint(N_samples)
        
        Psi_in = np.copy(Psi_in_dset[...,sample_index,:])
        
        target = np.copy(target_dset[...,sample_index,:])
        
        [Psi_echo,Theta_in,cost[step,:]] = HEB_step(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m)
        
        
    return [Theta_in,cost]



def optimization_av(Psi_in_dset,Theta_in,params,dt,N_steps,target_dset,eta,loss,m,N_train):
    
    # optimization loop 
    # implements N_train iterations of HEB
     
    [N_samples,N_traj] = np.shape(Psi_in_dset)[-2:] 
    
    cost = np.zeros((N_train,N_traj))
    
    for step in tqdm(range(N_train)):
        
        sample_index = np.random.randint(N_samples)
        
        Psi_in = np.copy(Psi_in_dset[...,sample_index,:])
        
        target = np.copy(target_dset[...,sample_index,:])
        
        [Psi_echo,Theta_in,cost[step,:]] = HEB_step_av(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m)
        
        
    return [Theta_in,cost]



def optimization_av_rec(Psi_in_dset,Theta_in,params,dt,N_steps,target_dset,eta,loss,m,N_train):
    
    # optimization loop 
    # implements N_train iterations of HEB
     
    [N_samples,N_traj] = np.shape(Psi_in_dset)[-2:] 
    
    cost = np.zeros((N_train,N_traj))
    
    output = [[],[],[],[]]
    
    for step in tqdm(range(N_train)):
        
        sample_index = np.random.randint(N_samples)
        
        Psi_in = np.copy(Psi_in_dset[...,sample_index,:])
        
        target = np.copy(target_dset[...,sample_index,:])
        
        [Psi_echo,Theta_in,cost[step,:],output_step] = HEB_step_av_rec(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m)
        
        output[sample_index].append(output_step)
        
        
    return [Theta_in,cost,output]




def optimization_mom(Psi_in_dset,Theta_in,params,dt,N_steps,target_dset,eta,loss,m,N_train):
    
    # optimization loop 
    # implements N_train iterations of HEB
     
    [N_samples,N_traj] = np.shape(Psi_in_dset)[-2:] 
    
    cost = np.zeros((N_train,N_traj))
    
    for step in tqdm(range(N_train)):
        
        sample_index = np.random.randint(N_samples)
        
        Psi_in = np.copy(Psi_in_dset[...,sample_index,:])
        
        target = np.copy(target_dset[...,sample_index,:])
        
        [Psi_echo,Theta_in,cost[step,:]] = HEB_step_mom(Psi_in,Theta_in,params,dt,N_steps,target,eta,loss,m)
        
        
    return [Theta_in,cost]
