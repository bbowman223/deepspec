import torch
import torch.nn as nn
import numpy as np
from hercules import autograd_hacks


def grad_size(model):
    """ Computes the number of parameters of a model

    :param model (torch.nn.Module): model to be parameter counted
    :return: number of (scalar) parameters
    """
    return sum([param.numel() for param in model.parameters()])


def coor_loss(output, cls):
    """ Auxilary function to extract gradients with respect to NN outputs.
        Computes a 'loss' to be the batch sum over a coordinate;
        This enables f_nabla_w for multi-output networks

    :param output (Tensor): the output of the NN of shape (batch_size X num_classes)
    :param cls: coordinate of output to be summed
    :return: summed coordinate output over a data batch
    """
    return torch.mean(output[:, cls])


def f_nabla_w(net_checkpoint, input_data):
    """ Computes the per-sample gradient of an NN outputs with respect to its weights for a given input_data batch
        Note this is different from computing the gradient of the NN **Loss** with respect to its weights.

        * To avoid any ambiguity on the model state, the gradient will be computed for the model parameters saved on net_checkpoint
        * for a model taking in input_data (batch_size, ...), and producing output (batch_size, num_outputs)
        this gradient will be of shape (batch_size, num_parameters, num_outputs)

    :param net (torch.nn.Module): Model over which to compute the gradient
    :param net_checkpoint (FILENAME): The parameter checkpoint to use for the model's parameters
    :param input_data (Tensor): The input data batch over which to compute the gradient
    :return: (Tensor) The per sample, per output gradients; of shape (batch_size, num_parameters, num_outputs)
    """

    net = torch.load(net_checkpoint)
    net.to('cpu')
    batch_size = input_data.size(0)
    num_classes = net(input_data).size(-1)
    param_size = grad_size(net)

    grad_matrix = np.empty((batch_size, param_size, num_classes))

    for cls in range(num_classes):
        net = torch.load(net_checkpoint)  # to avoid issues with auto_hacks
        net.to('cpu')
        net.train()

        autograd_hacks.add_hooks(net)
        output = net(input_data)
        autograd_hacks.clear_backprops(net)
        coor_loss(output, cls).backward(retain_graph=True)
        autograd_hacks.compute_grad1(net)
        autograd_hacks.disable_hooks()

        for example in range(batch_size):
            grad_matrix[example, :, cls] = np.concatenate(
                [param.grad1[example].flatten().cpu().numpy() for param in net.parameters()])

    return grad_matrix


def order_batch(data, targets, num_classes):
    """ Re-orders a batch of data by class label
        Now targets should be [0,0,0,... 1,1,1,..., 9,9,9...]

    :param data: data to be re-ordered
    :param targets: labels of the data-points, also to be re-ordered
    :param num_classes: the number of different potential classes
    :return: The re-ordered data, targets, and the range of indices in the batch for each class
    """

    data, targets = map(torch.stack, zip(*sorted(zip(data, targets), key = lambda x: x[1].item())))
    class_ranges = np.cumsum([0] + [np.count_nonzero(targets == i) for i in range(num_classes)])
    return data, targets, class_ranges


def NTK(grad_tensor, coor):
    """ Computes the NTK given a gradient f_nabla_w
        For multi-output networks, computes the NTK of a single output, f[coor]

    :param grad_tensor: per sample gradients with respect to outputs, of shape [batch, param, outputs]
    :param coor: Coordinate of output to compute the NTK over
    :return: The NTK of size [batch, batch]
    """
    if len(grad_tensor.shape) == 2:
        print('grad_tensor is a matrix, computing single-output NTK')
        grad_matrix_coor = grad_tensor
    else:
        grad_matrix_coor = grad_tensor[:,:,coor]
    return grad_matrix_coor.dot(grad_matrix_coor.T)


