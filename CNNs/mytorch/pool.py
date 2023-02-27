import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        out_height = A.shape[-1] - self.kernel + 1
        out_width = A.shape[-2] - self.kernel + 1
        self.A = A

        Z = np.zeros((A.shape[0], A.shape[1], out_width, out_height))

        self.max_indexes = np.zeros((A.shape[0], A.shape[1], out_width, out_height, 2))

        for batch in range(A.shape[0]):
            for in_channel in range(A.shape[1]):
                for j in range(A.shape[2]):
                    for i in range(A.shape[-1]):
                        if not(i+self.kernel>A.shape[-1] or j+self.kernel>A.shape[-2]):

                            slice = A[batch, in_channel, i:i+self.kernel, j:j+self.kernel]
                            Z[batch, in_channel,i,j] = np.max(slice)

                            ixes = np.array(list(np.unravel_index(slice.argmax(), slice.shape)))
                            self.max_indexes[batch, in_channel,i,j] = ixes + [i, j]


        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)

        for batch in range(dLdZ.shape[0]):
            for channel in range(dLdZ.shape[1]):
                for j in range(dLdZ.shape[-1]):
                    for i in range(dLdZ.shape[-2]):
                        ixes = self.max_indexes[batch, channel, i, j]
                        dLdA[batch, channel, int(ixes[0]), int(ixes[1])] += dLdZ[batch, channel, i, j]
        
        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        out_height = A.shape[-1] - self.kernel + 1
        out_width = A.shape[-2] - self.kernel + 1
        self.A = A

        Z = np.zeros((A.shape[0], A.shape[1], out_width, out_height))
        for batch in range(A.shape[0]):
            for in_channel in range(A.shape[1]):
                for j in range(A.shape[2]):
                    for i in range(A.shape[-1]):
                        if not(i+self.kernel>A.shape[-1] or j+self.kernel>A.shape[-2]):
                            slice = A[batch, in_channel, i:i+self.kernel, j:j+self.kernel]
                            
                            Z[batch, in_channel,i,j] = np.mean(slice)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = np.zeros(self.A.shape)
        for batch in range(dLdZ.shape[0]):
            for channel in range(dLdZ.shape[1]):
                for j in range(dLdZ.shape[-1]):
                    for i in range(dLdZ.shape[-2]):
                        shape1, shape2 = i+self.kernel, j+self.kernel
                        
                        dLdA[batch, channel, i:shape1, j:shape2] += dLdZ[batch, channel, i, j]/(self.kernel**2)

        return dLdA
class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z_stride_1 = self.maxpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z_stride_1)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA_big = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(dLdA_big)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z_stride_1 = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z_stride_1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA_big = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(dLdA_big)

        return dLdA
