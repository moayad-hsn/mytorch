from unittest import skip
import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
		
		Initialize instance variables

		Argument(s)
		-----------
		
		BLANK (int, optional): blank label index. Default 0.

		"""
        self.BLANK = BLANK
		# No need to modify
        


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
		"""


        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
            


        N = len(extended_symbols)


        skip_connect = []

        for ix in range(N):
            if ix+2 <=N-1:
                if extended_symbols[ix] != self.BLANK and extended_symbols[ix]!= extended_symbols[ix+2]:
                    skip_connect.append(1)
                else:
                    skip_connect.append(0)
            else:
                skip_connect.append(0)
        
        skip_connect.reverse()
                

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))
        
        return extended_symbols, skip_connect
        

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """


        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

		# -------------------------------------------->
		# TODO: Intialize alpha[0][0]
		# TODO: Intialize alpha[0][1]
		# TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
		# <---------------------------------------------
        alpha[0][0] = logits[0][extended_symbols[0]]
        alpha[0][1] = logits[0][extended_symbols[1]]

        for t in range(1, T):
            alpha[t][0] = logits[t][0]*alpha[t-1][0]
            for s in range(1, S):
                symb = extended_symbols[s]
                alpha[t][s] = alpha[t-1][s] + alpha[t-1][s-1]
                if skip_connect[s] == 1:
                    alpha[t][s] += alpha[t-1][s-2]
                alpha[t][s] = alpha[t][s] * logits[t][symb]
        
        return alpha


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
		
		"""

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))
        betahat = np.zeros(shape=(T, S))
        skip_connect_back = list(skip_connect)
        skip_connect_back.reverse()

        betahat[-1][-1] = logits[-1][extended_symbols[-1]]
        betahat[-1][-2] = logits[-1][extended_symbols[-2]] 

        for t in reversed(range(T-1)):
            betahat[t][-1] = betahat[t+1][-1]*logits[t][extended_symbols[-1]]
            for s in reversed(range(S-1)):
                betahat[t][s] = betahat[t+1][s] + betahat[t+1][s+1]
                if skip_connect_back[s] == 1:
                    betahat[t][s]+=betahat[t+1][s+2]
                betahat[t][s]= betahat[t][s] * logits[t][extended_symbols[s]]
        
        
        for t in reversed(range(T)):
            for s in reversed(range(S)):
                if logits[t][extended_symbols[s]]!=0:
                    beta[t][s] = betahat[t][s] / logits[t][extended_symbols[s]]

            

        return beta
            

		

		

		# -------------------------------------------->
		# TODO
		# <--------------------------------------------

		# return beta
		

    def get_posterior_probs(self, alpha, beta):

        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

		"""


        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        for t in range(T):
            sumgamma[t] = 0
            for s in range(S):
                gamma[t][s] = alpha[t][s] * beta[t][s]
                sumgamma[t]+=gamma[t][s]
            for s in range(S):
                gamma[t][s] = gamma[t][s] / sumgamma[t]
        
        return gamma
		

		

		# -------------------------------------------->
		# TODO
		# <---------------------------------------------

		# return gamma
		

class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.
        
		"""
		# -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
		# <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []
        new_targets = []
        new_logits = []

        for batch_itr in range(B):
           # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            new_targets = target[batch_itr, :target_lengths[batch_itr]]
            #     Truncate the logits to input length
            new_logits = logits[:input_lengths[batch_itr], batch_itr]
            #     Extend target sequence with blank
            self.extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target=new_targets)
            #     Compute forward probabilities
            alphas = self.ctc.get_forward_probs(new_logits, self.extended_symbols, skip_connect)
            #     Compute backward probabilities
            betas = self.ctc.get_backward_probs(new_logits, self.extended_symbols, skip_connect)
            #     Compute posteriors using total probability function
            gammas = self.ctc.get_posterior_probs(alphas, betas)
            #     Compute expected divergence for each batch and store it in totalLoss

            # div(gammas, new logits)??
            # print(new_logits.shape, gammas.shape)
            for i in range(new_logits.shape[0]):
                for ix, symbol in enumerate(self.extended_symbols):
                    total_loss[batch_itr] -= (np.log(new_logits[i][symbol])*gammas[i][ix]) ## Redo this with multiply
            

            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            # pass

        total_loss = np.sum(total_loss) / B
        self.total_loss = total_loss
		
        return total_loss
        raise NotImplementedError
		

    def backward(self):
        """
		
		CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
		w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            new_targets = self.target[batch_itr, :self.target_lengths[batch_itr]]
            #     Truncate the logits to input length
            new_logits = self.logits[:self.input_lengths[batch_itr], batch_itr]
            #     Extend target sequence with blank
            self.extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target=new_targets)
            alphas = self.ctc.get_forward_probs(new_logits, self.extended_symbols, skip_connect)
            betas = self.ctc.get_backward_probs(new_logits, self.extended_symbols, skip_connect)
            gammas = self.ctc.get_posterior_probs(alphas, betas)
            #     Compute derivative of divergence and store them in dY
            for seq in range(new_logits.shape[0]):
                for ix, symbol in enumerate(self.extended_symbols):
                    dY[seq, batch_itr, symbol] -= gammas[seq][ix]/new_logits[seq][symbol]
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            # pass

        return dY
        raise NotImplementedError
