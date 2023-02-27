# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

from matplotlib.pyplot import axes
import numpy as np
from torch import conv1d
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
            W : (out_channels, in_channels, kernel_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        out_size = (np.array(A.shape[-1]) - self.W.shape[-1]) + 1
        self.input_size = A.shape[-1]
        Z = np.zeros((self.A.shape[0], self.out_channels, out_size))
        
        for batch in range(self.A.shape[0]):
            for out_channel in range(self.out_channels):
                for i in range(out_size):
                    if not(i+self.kernel_size > self.A.shape[-1]):
                        val = np.sum(np.multiply(self.A[batch, :, i:i+self.kernel_size], self.W[out_channel, :, :])) 
                        Z[batch, out_channel, i] = val+ self.b[out_channel]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
            
        flipped_w = np.flip(self.W, axis=(-1))
        
        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        pad_width = self.kernel_size - 1
        padded_dLdZ = np.pad(
            dLdZ,
            ((0, 0), (0, 0), (pad_width, pad_width)),
            "constant",
            constant_values=0,
        )
        
        for batch in range(self.A.shape[0]):
            for out_channel in range(self.out_channels):
                for in_channel in range(self.in_channels):
                    for i in range(self.input_size):
                        if not(i+dLdZ.shape[-1] > self.A.shape[-1]):
                            val = np.sum(self.A[batch, in_channel, i:i+dLdZ.shape[-1]]* dLdZ[batch, out_channel, :])
                            self.dLdW[out_channel, in_channel, i] += val
        
        dLdA = np.zeros((self.A.shape[0], self.in_channels, self.input_size))
        for batch in range(self.A.shape[0]):
            for in_channel in range(self.in_channels):
                for i in range(self.input_size):
                    if not(i+self.kernel_size > padded_dLdZ.shape[-1]):
                        val =  np.sum(padded_dLdZ[batch, :, i:i+self.kernel_size]* flipped_w[:,in_channel, :])
                        dLdA[batch, in_channel, i] += val
        return dLdA

        

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                 weight_init_fn, bias_init_fn) # TODO
        self.downsample1d = Downsample1d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z_stride1 = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z_stride1) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ_intermediate = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ_intermediate) # TODO 

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        sizes = tuple(np.array(self.A.shape[-2:])-np.array(self.kernel_size) + 1)
        Z = np.zeros((self.A.shape[0], self.out_channels, sizes[0], sizes[1])) #TODO
        self.z_shape = Z.shape

        for batch in range(self.A.shape[0]):
            for out_channel in range(self.out_channels):
                for in_channel in range(self.in_channels):
                    for j in range(sizes[1]):
                        for i in range(sizes[0]):
                            Z[batch, out_channel, i, j] += np.sum(self.A[batch, in_channel, i:i+self.kernel_size,j:j+self.kernel_size] * 
                            self.W[out_channel, in_channel, :, :])
                Z = Z + +self.b[out_channel]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        sizes = tuple(np.array(dLdZ.shape[-2:])+np.array(self.kernel_size) - 1)

        #self.dLdW = None # TODO        
        for batch in range(self.A.shape[0]):
            for out_channel in range(self.out_channels):
                for in_channel in range(self.in_channels):
                    for j in range(self.kernel_size):
                        for i in range(self.kernel_size):
                            self.dLdW[out_channel, in_channel, i, j] += np.sum(self.A[batch, in_channel, i:i+dLdZ.shape[-2],j:j+dLdZ.shape[-1]] 
                            * dLdZ[batch, out_channel, :, :])


        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3)) # TODO

        pad_width = self.kernel_size - 1
        padded_dLdZ = np.pad(
            dLdZ,
            ((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)),
            "constant",
            constant_values=0,
        )

        flipped_w = np.flip(self.W, axis=(-1, -2))
        
        
        dLdA = np.zeros(self.A.shape) # TODO

        for batch in range(padded_dLdZ.shape[0]):
            for in_channel in range(self.in_channels):
                for out_channel in range(self.out_channels):
                    for j in range(dLdA.shape[-1]):
                        for i in range(dLdA.shape[-2]):
                            dLdA[batch, in_channel, i, j] += np.sum(padded_dLdZ[batch, out_channel, i:i+flipped_w.shape[-2],j:j+flipped_w.shape[-1]] 
                            * flipped_w[out_channel, in_channel, :, :])
        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 =  Conv2d_stride1(in_channels, out_channels,kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample2d = Downsample2d(self.stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z_1stride = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z_1stride) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdA_stride_1 = self.downsample2d.backward(dLdZ)
        
        

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA_stride_1) # TODO

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(self.upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA =  self.upsample1d.backward(delta_out) #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(self.upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA =  self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.shapes = A.shape
        Z = A.reshape((A.shape[0], -1)) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = dLdZ.reshape(self.shapes) #TODO

        return dLdA

