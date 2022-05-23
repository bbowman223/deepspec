import torch
from shallow import Shallow, KernelCreator, Activations
import matplotlib.pyplot as plt

import numpy as np
import scipy.linalg
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class FigureGen:
    def get_normalized_spectrum(
        self,
        xs,
        input_d,
        batch_size,
        width,
        suptitle
    ):
        act  =  Activations.softplus
        d_act = Activations.d_softplus
        act = np.vectorize(act)
        d_act = np.vectorize(d_act)

        fig, axs = plt.subplots() 
        
        net = Shallow(width,
                      input_d,
                      act,
                      d_act,
                      True)

        creator = KernelCreator(net)
        ker = creator.k_total(xs)

        eigs, dirs = scipy.linalg.eigh(ker)
        eigs = np.flip(eigs)
        eigs /= max(eigs)

        return eigs 


    def gen_fig(self, batch_size, width, num_runs):
        data_train = CIFAR10('./data/cifar',
                             train=True,
                             download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: torch.flatten(x))]))
        data_train_loader = DataLoader(data_train,
                                batch_size=batch_size,
                                shuffle=True) 


        fig, axs = plt.subplots()
        axs.set_yscale('log')
   
        for i in range(num_runs):
            xs, ys = next(iter(data_train_loader))
            xs = xs.numpy().T
            input_d = xs.shape[0]

            spec = self.get_normalized_spectrum(
                xs,
                input_d, 
                batch_size,
                width,
                'NTK Spectrum on CIFAR10'
            ) 

            axs.plot(spec[:batch_size//2])

        fig.set_size_inches(8, 6)
        fig.savefig('shallow_cifar.png')

if __name__ == '__main__':
    figgen = FigureGen()
    figgen.gen_fig(2000, 4000, 10)


