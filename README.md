# self_learning_machines
Code for 'Self-learning Machines based on Hamiltonian Echo Backpropagation'


Code for the manuscript 'Self-learning Machines based on Hamiltonian Echo Backpropagation', Victor Lopez-Pastor, Florian Marquardt:
https://arxiv.org/abs/2103.04992

Abstract
 A physical self-learning machine can be defined as a nonlinear dynamical system that can be trained on data (similar to artificial neural networks), but where the update of the internal degrees of freedom that serve as learnable parameters happens autonomously. In this way, neither external processing and feedback nor knowledge of (and control of) these internal degrees of freedom is required. We introduce a general scheme for self-learning in any time-reversible Hamiltonian system. We illustrate the training of such a self-learning machine numerically for the case of coupled nonlinear wave fields. 
 
 This code enables you to reproduce the following figures displayed on the paper: Figure 7, Figure 8. 
 
 The folders provided here are structured as follows:
 
 - mnist_photonic_cnn contains the code required to reproduce the numerical simulations of the training a photonic neural network on MNIST (right hand side of figure 7).
 - XOR_wavefields contains the code required to reproduce the numerical simulations of the self-learning device based on nonlinear wavefields (left hand side of figure 7).
 - cavity_array contains the code required to reproduce the numerical simulations of the self-learning device based on a nonlinear cavity array, except the case that includes disorder (figure 8). 
 - cavity_array_disorder contains the code required to reproduce the numerical simulations of the self-learning device based on a nonlinear cavity array with disorder (figure 8)c).
