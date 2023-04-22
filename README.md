This repository contains the code for the paper 
["Spectral Bias Outside the Training Set for Deep
Networks in the Kernel Regime"](https://arxiv.org/abs/2206.02927) (NeurIPS 2022).  If you use this code in a paper, please cite this paper using the bibtex reference below
```
@inproceedings{NEURIPS2022_c4006ff5,
 author = {Bowman, Benjamin and Montufar, Guido F},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {30362--30377},
 publisher = {Curran Associates, Inc.},
 title = {Spectral Bias Outside the Training Set for Deep Networks in the Kernel Regime},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/c4006ff54a7bbda74c09bad6f7586f5b-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```

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

The dependencies for the project are specified in the **environment.yml** file.
If you use [Anaconda](https://www.anaconda.com/) you can construct the environment from this file.

Acknowledgements:
We would like to thank [Yonatan Dukler](https://github.com/dukleryoni) for sharing portions of this code during a project in 2019.
The code to compute the NTK Gram matrix in PyTorch is powered by [autograd-hacks](https://github.com/cybertronai/autograd-hacks)
