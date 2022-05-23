import numpy as np
import scipy.linalg
import unittest


class Activations:
    '''
    Class with relevant activation functions.
    '''


    @staticmethod
    def relu(x):
        return max(0.0, x)


    @staticmethod
    def d_relu(x):
        return 1.0 if x > 0.0 else 0.0


    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))


    @staticmethod
    def d_sigmoid(x):
        return Activations.sigmoid(x) * (1.0 - Activations.sigmoid(x))


    @staticmethod
    def tanh(x):
        return np.tanh(x)


    @staticmethod
    def d_tanh(x):
        return 1.0 - np.square(np.tanh(x)) 


    @staticmethod
    def softplus(x):
        return np.log(1.0 + np.exp(x)) 


    @staticmethod
    def d_softplus(x):
        return Activations.sigmoid(x)


class Shallow:
    '''
    Class representing a shallow neural network
    '''


    def __init__(
        self,
        width,
        input_d,
        activation,
        d_activation,
        unit_outer = False
    ):
        '''
        Initializes a shallow neural network with random parameters

        Parameters
        :width (int): the width of the network
        :input_d (int): the input dimension of the network
        :activation (callable):
            Vectorized function representing activation function
        :d_activation (callable): Scalar function representing
            derivative of activation function
        :unit_outer (bool): True if we should force the outer
            layer weights to be unit magnitude, False otherwise
        '''


        self.width = width
        self.input_d = input_d
        self.inner_ws = np.random.normal(size=(width, input_d))
        self.inner_bs = np.random.normal(size=width)
        self.outer_ws = np.random.normal(size=width)
        if unit_outer:
            f = lambda x: 1.0 if x >= 0.0 else -1.0
            f = np.vectorize(f)
            self.outer_ws = f(self.outer_ws)
        outer_b  = np.random.normal()
        self.activation = activation
        self.d_activation = d_activation


    def hidden_features(self, xs, nonlinearity = None):
        '''
        Computes the features provided by the hidden layer
        for a given set of inputs and entrywise nonlinearity.
        Specifically will compute nonlinearity applied entrywise
        to (np.matmul(self.inner_ws, xs) + self.inner_bs)

        Parameters
        :xs (np.array):
            Array of shape (input_dim, k) k arbitrary representing k inputs
        :nonlinearity (callable):
            Nonlinearity to apply entrywise to linear function of hidden layer.
            Set to self.activation by default 

        Returns
        :features (np.array):
            Array of shape (network width, k) representing hidden layer features
        '''

        if nonlinearity is None:
            nonlinearity = self.activation

        return nonlinearity(np.matmul(self.inner_ws, xs).T + self.inner_bs).T


    def hidden_grad_features(self, xs):
        '''
        Computes the features for the hidden layer with the derivative
        of the activation applied entrywise instead of the activation itself

        Parameters
        :xs (np.array):
            Array of shape (input_dim, k) k arbitrary representing k inputs

        Returns
        :features (np.array):
            Array of shape (network width, k) representing hidden layer features
        '''

        return self.hidden_features(xs, self.d_activation)


    def forward(self, xs):
        '''
        Computes the outputs of the network applied to an array of inputs

        Parameters
        :xs (np.array): Array of shape (input_dim, k) for k arbitrary
            representing k inputs
        
        Returns
        :outputs (np.array): 1D Array of size k representing the k outputs
        '''  

        return((1.0 / np.sqrt(self.width)) *
            np.dot(self.outer_ws, self.hidden_features(xs))
            + self.outer_b)


    def set_params(
        self,
        inner_ws,
        inner_bs,
        outer_ws,
        outer_b,
        activation,
        d_activation
    ):
        '''
        Updates the network to have the specified parameters

        Parameters
        :inner_ws (np.array): Array of shape (network width, input dimension)
            representing hidden layer weights
        :inner_bs (np.array): 1D-Array of size 'network width'
            representing inner layer biases
        :outer_ws (np.array): 1D-Array of size 'network width'
            representing outer layer weights
        :outer_b (np.double): Outer layer bias
        :activation (callable):
            Vectorized function representing activation function
        :d_activation (callable): Scalar function representing
            derivative of activation function
        '''

        self.inner_ws = inner_ws
        self.inner_bs = inner_bs
        self.outer_ws = outer_ws
        self.outer_b = outer_b
        self.activation = activation
        self.d_activation = d_activation
        self.width = inner_ws.shape[0]
        self.input_dim = inner_ws.shape[1]
        

class KernelCreator:
    '''
    Class for computing the kernels associated with a shallow network
    '''


    def __init__(self, net):
        '''
        Initializes the kernel creator with an associated neural net.

        Parameters
        :net (Shallow): the neural net associated with the creator
        '''

        self.net = net


    def k_inner(self, xs):
        '''
        Returns the gram matrix associated with the kernel
        corresponding to the hidden layer.

        Parameters
        :xs (np.array): numpy array of shape (input_dim, # samples)
            associated with the inputs

        Returns
        :gram (np.array): numpy array of shape (# samples, # samples)
            representing gram matrix associated with kernel
        '''

        data_gram = np.matmul(xs.T, xs)
        feat = (self.net.hidden_grad_features(xs).T * self.net.outer_ws).T
        inner_gram = np.matmul(feat.T, feat)
        scale = 1.0 / self.net.width
        return(scale*inner_gram*(data_gram + np.ones_like(data_gram)))

    
    def k_outer(self, xs):
        '''
        Returns the gram matrix associated with the kernel
        corresponding to the outer layer 

        Parameters
        :xs (np.array): numpy array of shape (input_dim, # samples)
            associated with the inputs

        Returns
        :gram (np.array): numpy array of shape (# samples, # samples)
            representing gram matrix associated with kernel
        '''

        feat = self.net.hidden_features(xs)
        return(np.matmul(feat.T, feat) / self.net.width)


    def k_total(self, xs, outer_bias=True):
        '''
        Returns the gram matrix associated with the kernel corresponding
        to all parameters of the network.

        Parameters
        :xs (np.array): numpy array of shape (input_dim, # samples)
            associated with the inputs
        :outer_bias (bool): Boolean indicating if the network has an outer
            bias term.

        Returns
        :gram (np.array): numpy array of shape (# samples, # samples)
            representing gram matrix associated with kernel
        '''

        return (self.k_inner(xs) + self.k_outer(xs) +
                int(outer_bias) * np.ones((xs.shape[-1], xs.shape[-1])))
        
        
class TestKernel(unittest.TestCase):
    '''
    Class for testing KernelCreator
    '''
    @classmethod
    def setUpClass(cls):
        cls.width = 100000
        cls.input_d = 1000
        cls.n = 100
        cls.activation = np.vectorize(Activations.relu) 
        cls.d_activation = np.vectorize(Activations.d_relu)
        cls.net = Shallow(cls.width,
                          cls.input_d,
                          cls.activation,
                          cls.d_activation,
                          True)


    def test_k_inner(self):
        '''
        Tests function KernelCreator.k_inner by comparing against
        explicit form
        '''

        xs = np.random.rand(TestKernel.input_d, TestKernel.n)
        ker_creator = KernelCreator(TestKernel.net)
        ker1 = ker_creator.k_inner(xs)

        
        ker2 = np.zeros_like(ker1)
        for i in range(TestKernel.n):
            for j in range(i, TestKernel.n):
                ker2[i][j] = self._cos_inner_ker(xs[:,i], xs[:,j])
                ker2[j][i] = ker2[i][j]


        err = 2*np.mean(abs(ker1 - ker2)) / (np.mean(ker1) + np.mean(ker2))

        self.assertTrue(err < .01)

    
    def test_k_outer(self):
        '''
        Tests function KernelCreator.k_outer by comparing against explicit
        form
        '''

        xs = np.random.rand(TestKernel.input_d, TestKernel.n)
        ker_creator = KernelCreator(TestKernel.net)
        ker1 = ker_creator.k_outer(xs)

        ker2 = np.zeros_like(ker1)

        for i in range(TestKernel.n):
            for j in range(i, TestKernel.n):
                ker2[i][j] = self._cos_outer_ker(xs[:,i], xs[:,j])
                ker2[j][i] = ker2[i][j]

        err = 2*np.mean(abs(ker1 - ker2)) / (np.mean(ker1) + np.mean(ker2))

        self.assertTrue(err < .01)


    def test_k_total(self):
        '''
        Tests function KernelCreator.k_total by comparing against explicit
        form
        '''

        outer_bias = False
        xs = np.random.rand(TestKernel.input_d, TestKernel.n)
        ker_creator = KernelCreator(TestKernel.net)
        ker1 = ker_creator.k_total(xs, outer_bias)
        
        ker2 = np.zeros_like(ker1)
        for i in range(TestKernel.n):
            for j in range(i, TestKernel.n):
                ker2[i][j] = (self._cos_outer_ker(xs[:,i], xs[:,j]) + 
                              self._cos_inner_ker(xs[:,i], xs[:,j]))
                ker2[j][i] = ker2[i][j]

        err = 2*np.mean(abs(ker1 - ker2)) / (np.mean(ker1) + np.mean(ker2))
        self.assertTrue(err < .01)

        outer_bias = True
        ker1 = ker_creator.k_total(xs, outer_bias)
        ker2 += np.ones_like(ker2)

        err = 2*np.mean(abs(ker1 - ker2)) / (np.mean(ker1) + np.mean(ker2))
        self.assertTrue(err < .01)


          
    def _cos_inner_ker(self, x, y):
        '''
        Implements explicit form of the cosine kernel associated with the
        hidden layer of a shallow relu network.
        See "Kernel Methods for Deep Learning" by Youngmin Cho, Lawrence Saul
        NIPS 2009
        https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html

        Parameters
        :x (np.array): 1D array representing an input to the kernel
        :y (np.array): 1D array representing an input to the kernel 
        '''

        inner_prod = np.dot(x, y)
        nx = np.sqrt(np.dot(x, x) + 1)
        ny = np.sqrt(np.dot(y, y) + 1)
        acos_arg = (inner_prod + 1) / (nx * ny)
        if abs(acos_arg) > 1.0:
            acos_arg = 1.0 if acos_arg > 1.0 else -1.0

        return((inner_prod + 1)*(np.pi - np.arccos(acos_arg)) / (2.0 * np.pi))


    def _cos_outer_ker(self, x, y):
        '''
        Implements explicit form of the cosine kernel associated with the
        outer layer of a shallow relu network
        See "Kernel Methods for Deep Learning" by Youngmin Cho, Lawrence Saul
        NIPS 2009
        https://papers.nips.cc/paper/2009/hash/5751ec3e9a4feab575962e78e006250d-Abstract.html

        Parameters
        :x (np.array): 1D array representing an input to the kernel
        :y (np.array): 1D array representing an input to the kernel 
        '''

        inner_prod = np.dot(x, y)
        nx = np.sqrt(np.dot(x, x) + 1)
        ny = np.sqrt(np.dot(y, y) + 1)
        acos_arg = (inner_prod + 1) / (nx * ny)
        if abs(acos_arg) > 1.0:
            acos_arg = 1.0 if acos_arg > 1.0 else -1.0
        theta = np.arccos(acos_arg)

        return(nx*ny*(np.sin(theta) + (np.pi - theta)*np.cos(theta))/(2.0 * np.pi))


if __name__ == '__main__':
    unittest.main() 
