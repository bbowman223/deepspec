This project is organized into two directories.
* **deep/**
    * Computes the NTK spectrum for LeNet-5 using Pytorch
* **shallow/**
    * Computes the NTK spectrum for a shallow network from scratch

A natural question to ask is why not compute the NTK for both networks
using PyTorch instead of implementing the shallow model from scratch?
The primary reason was that we already had the code for both cases written
beforehand so we figured we might as well use our existing code.
The benefit of the shallow implementation is it is more readable and
transparent how the computation is being performed whereas the
PyTorch implementation requires some hacks to compute the NTK using autograd.
Furthermore the shallow case can be tested against closed formulas and has
unit tests written whereas the deep case does not.

The directories **deep/** and **shallow/** both have their own **README** files to explain
the project organization
