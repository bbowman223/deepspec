This directory contains the files to compute the NTK spectrum for LeNet-5 on MNIST.
If you want to create the figure plotting the NTK spectrum go to the directory notebooks/
and open the jupyter notebook Plot_NTK_Spectrum.ipynb

Below is an explanation of each directory

**LeNet/**

Defines the network model

**checkpoints/**

Where you store checkpoints (https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
for the network.  The code to compute the NTK computes the NTK for a network using its checkpoint

**data/**

Directory to store the MNIST data

**hercules/**

This is where all the heavylifting occurs.
autograd_hacks.py is a file that makes hacks to autograd
to allow for computation of the gradients of the model
with respect to a particular input.  This part of the project
is thanks to (https://github.com/cybertronai/autograd-hacks)
ntk_utils.py contains the code to compute the NTK from a
checkpoint

**notebooks/**

Contains the jupyter notebook to generate the figure to plot
the NTK spectrum
