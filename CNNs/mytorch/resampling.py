import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        Z = np.kron(A, np.array([1] +[0]*(self.upsampling_factor-1)))

        Z = Z[:, :, :(A.shape[-1]*self.upsampling_factor)-(self.upsampling_factor-1)] # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        
        dLdA = dLdZ[:, :, ::self.upsampling_factor]  #TODO

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.A_shape = A.shape

        Z = A[:, :, ::self.downsampling_factor] # TODO

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        dLdA = np.kron(dLdZ, np.array([1] +[0]*(self.downsampling_factor-1)))

        dLdA= dLdA[:, :, :self.A_shape[-1]:] # TODO


        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        kron_array = [[1]+[0]*(self.upsampling_factor-1), *[[0]*self.upsampling_factor]*(self.upsampling_factor-1) ]

        Z = np.kron(A, kron_array) # TODO

        return Z[:, :, :(A.shape[-2]*self.upsampling_factor)-(self.upsampling_factor-1), :(A.shape[-1]*self.upsampling_factor)-(self.upsampling_factor-1)]

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA = dLdZ[:, :, ::self.upsampling_factor, ::self.upsampling_factor]  #TODO

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        self.A_size = A.shape
        Z = A[:, :, ::self.downsampling_factor, ::self.downsampling_factor]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        kron_array = [[1]+[0]*(self.downsampling_factor-1), *[[0]*self.downsampling_factor
        ]*(self.downsampling_factor-1) ]

        dLdA = np.kron(dLdZ, kron_array) # TODO
        dLdA = dLdA[:, :, :self.A_size[-2], :self.A_size[-1]]  #TODO

        return dLdA