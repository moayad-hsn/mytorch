import numpy as np


def compress(path):
    
    path = str(path).replace("'","")
    path = path.replace(",","")
    path = path.replace(" ","")
    path = path.replace("[","")
    path = path.replace("]","")
    
    return path


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """
        _, seq_len, _ = y_probs.shape


        decoded_path = []
        blank = 0
        path_prob = 1

        for seq in range(seq_len):
            sequence_probs = y_probs[:, seq, 0]
            path_prob *= np.max(sequence_probs)
            index = np.argmax(sequence_probs)
            symbol = " " if (index == blank) else self.symbol_set[index-1]
            
            if seq>0 and symbol == compress(decoded_path)[-1]:
                symbol = " "
            
            decoded_path.append(symbol)  
        decoded_path = compress(decoded_path)

        return decoded_path, path_prob
        


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
    
    def extend_blank(self, terminal_blank, terminal_symbol, blank_score, path_score, y_prob):
        new_blank_scores = {}
        new_blank_path = []
        for path in set(terminal_blank):
            new_blank_path.append(path)
            new_blank_scores[path] = blank_score[path]*y_prob[0]
        for symbol in set(terminal_symbol):
            if symbol in new_blank_path:
                new_blank_scores[symbol] += path_score[symbol]*y_prob[0]
            else:
                new_blank_path.append(symbol)
                new_blank_scores[symbol] = path_score[symbol]*y_prob[0]
        return new_blank_path, new_blank_scores 
    
    def extend_symbol(self, blank_path, symbol_path, blank_score, path_score, y_prob, symbols_set):
        new_path_score = {}
        new_symbol_path = set()
        for blank in blank_path:
            for ix in range(len(symbols_set)): 
                padded_symb = blank + symbols_set[ix]
                new_symbol_path.add(padded_symb)
                new_path_score[padded_symb] = blank_score[blank] * y_prob[ix+1]

        for symbols in symbol_path:
            for ix in range(len(symbols_set)): 
                padded_symb = symbols if (symbols_set[ix] == symbols[-1]) else symbols + symbols_set[ix] 
                if padded_symb in new_symbol_path: 
                    new_path_score[padded_symb] += path_score[symbols] * y_prob[ix+1]
                else: 
                    new_symbol_path.add(padded_symb)
                    new_path_score[padded_symb] = path_score[symbols] * y_prob[ix+1]
        return new_symbol_path, new_path_score
    
    def prune(self, terminal_blank, terminal_symbol, blank_score, path_score, beam):
        prune_blank_score = {}
        final_blank_path = set()
        prune_symbol_score = {}
        final_symbol_path = set()
        all_scores = []
        
        for blank in terminal_blank:
            all_scores.append(blank_score[blank])
        for symbol in terminal_symbol:
            all_scores.append(path_score[symbol])

        all_scores.sort(reverse=True)

        last_score = all_scores[min(beam, len(all_scores)-1)]

        for blank in terminal_blank:
            if blank_score[blank] > last_score:
                final_blank_path.add(blank)
                prune_blank_score[blank] = blank_score[blank]

        for symbol in terminal_symbol:
            if path_score[symbol] > last_score:
                final_symbol_path.add(symbol)
                prune_symbol_score[symbol] = path_score[symbol]


        return final_blank_path, final_symbol_path, prune_blank_score, prune_symbol_score

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None

        FinalPathScore = {}
        blank_scores = {}
        blanks = set()

        for i in range(len(self.symbol_set)):
            FinalPathScore[self.symbol_set[i]] = y_probs[:,0][i + 1]        
        blank_scores[""] = y_probs[:,0][0] 
        blanks.add("")

        all_symbols = self.symbol_set

        for seq in range(1, y_probs.shape[1]):
            temp_blank_path, temp_symbol_path, temp_blank_scores, temp_path_scores = self.prune(blanks, all_symbols,blank_scores, FinalPathScore, self.beam_width)

            blanks, blank_scores =  self.extend_blank(temp_blank_path, temp_symbol_path,  temp_blank_scores, temp_path_scores, y_probs[:, seq])

            all_symbols, FinalPathScore = self.extend_symbol(temp_blank_path, temp_symbol_path, temp_blank_scores, temp_path_scores, y_probs[:, seq], self.symbol_set)

        for symbol in blanks:
            if symbol in all_symbols:
                FinalPathScore[symbol] += blank_scores[symbol]

        bestPath = max(FinalPathScore, key=FinalPathScore.get)

        return bestPath, FinalPathScore
        
        
        #return bestPath, FinalPathScore


        raise NotImplementedError
