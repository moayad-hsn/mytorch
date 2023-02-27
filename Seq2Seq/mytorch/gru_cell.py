import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h):
        return self.forward(x, h)

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx

        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh

        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx

        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def forward(self, x, h):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.r_z = self.Wrx.dot(x) +self.brx + self.Wrh.dot(h) + self.brh
        self.r = self.r_act.forward(self.r_z)

        self.z_z = self.Wzx.dot(x) +self.bzx + self.Wzh.dot(h) + self.bzh
        self.z = self.z_act.forward(self.z_z)

        self.n_z = self.Wnx.dot(x) + self.bnx + self.r * (self.Wnh.dot(h) + self.bnh)
        self.n = self.h_act.forward(self.n_z)

        h_t = (1-self.z)*self.n + self.z*h

        
        # This code should not take more than 10 lines. 
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t
        # raise NotImplementedError

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        x = self.x.reshape(self.d, 1)
        h = self.hidden.reshape(self.h, 1)

        dn = delta * (1-self.z) * self.h_act.backward()

        dr = dn * (self.Wnh.dot(self.hidden) + self.bnh) * self.r_act.backward()

        dz = delta * (self.hidden - self.n) * self.z_act.backward()


        self.dWnx = np.dot(x, dn).T
        self.dbnx = dn
        temp_dn = dn * self.r
        self.dWnh = np.dot(h, temp_dn).T
        self.dbnh = temp_dn

        self.dWrx = np.dot(x, dr).T
        self.dbrx = dr
        self.dWrh = np.dot(h, dr).T
        self.dbrh = dr

        self.dWzx = np.dot(x, dz).T
        self.dbzx = dz
        self.dWzh = np.dot(h, dz).T
        self.dbzh = dz

        dx = np.dot(dr, self.Wrx) + np.dot(dn, self.Wnx) + np.dot(dz, self.Wzx)

        dh = np.dot(dr, self.Wrh) + np.dot(dz, self.Wzh) + + np.multiply(delta, self.z) + np.dot(dn*self.r, self.Wnh)

        dx = dx.reshape(1, self.d)
        dh = dh.reshape(1, self.h)


        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
    