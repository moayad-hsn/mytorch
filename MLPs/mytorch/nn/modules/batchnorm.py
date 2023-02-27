import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):
        
        self.alpha     = alpha
        self.eps       = 1e-8
        
        self.Z         = None
        self.NZ        = None
        self.BZ        = None

        self.BW        = np.ones((1, num_features))
        self.Bb        = np.zeros((1, num_features))
        self.dLdBW     = np.zeros((1, num_features))
        self.dLdBb     = np.zeros((1, num_features))
        
        self.M         = np.zeros((1, num_features))
        self.V         = np.ones((1, num_features))
        
        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        
        if eval:
            # TODO
            self.NZ = (Z - self.running_M)/((self.running_V + self.eps)**0.5)
            self.BZ = np.multiply(self.NZ, self.BW) + self.Bb
            return self.BZ
            
        self.Z         = Z
        self.N         = Z.shape[0] # TODO
        
        self.M         = (1/self.N) * np.sum(Z, axis=0, keepdims= True) # TODO
        self.V         = (1/self.N) * np.sum((Z - self.M)**2, axis= 0, keepdims= True) # TODO
        self.NZ        = (Z - self.M)/np.sqrt(self.V + self.eps) # TODO
        self.BZ        = self.NZ * self.BW + self.Bb # TODO
        
        self.running_M = self.alpha * self.running_M + (1 - self.alpha)*self.M # TODO
        self.running_V = self.alpha * self.running_V + (1 - self.alpha)*self.V # TODO
        
        return self.BZ

    def backward(self, dLdBZ):
        
        self.dLdBW  = np.sum(dLdBZ * self.NZ, axis= 0, keepdims= True) # TODO
        self.dLdBb  = np.sum(dLdBZ, axis= 0, keepdims= True) # TODO
        
        dLdNZ       = dLdBZ * self.BW  # TODO
        dLdV        = (-0.5)*np.sum(np.multiply(dLdNZ, (self.Z - self.M)*((self.V + self.eps)**(-1.5))), 
                                         axis= 0, keepdims= True)# TODO
        dLdM1       = -np.sum(dLdNZ/((self.V + self.eps)**0.5), axis= 0, keepdims= True)
        dLdM2       = (-2/dLdBZ.shape[0])*np.multiply(dLdV, np.sum((self.Z - self.M), axis=0, keepdims= True))
        dLdM        = dLdM1 + dLdM2 # TODO
        
        dLdZ        = dLdNZ/((self.V + self.eps)**0.5) + (2/dLdBZ.shape[0])*dLdV*(self.Z - self.M) + (1/dLdBZ.shape[0])*dLdM
        
        return  dLdZ