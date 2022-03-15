import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

class Helper(object):
    def __init__(self, X, y):
        super(Helper, self).__init__()
        self.w2idx, self.idx2w = self.get_mapper(X, params=['<PAD>', '<BEG>', '<END>'])
        self.tag2idx, self.idx2tag = self.get_mapper(y, params=['<PAD>', '<BEG>', '<END>'])

    def get_mapper(self, X, params=None) -> tuple:
        '''
        return w2idx, idx2w
        '''
        xx = []
        for _x in X:
            xx.extend(_x)
        xx = np.unique(xx)
        # words = ['<PAD>', '<BEG>', '<END>'] + xx
        words = params if params is not None else []
        words.extend(xx)
        w2idx = {word: idx for idx, word in enumerate(words)}
        idx2w = {idx: word for idx, word in enumerate(words)}
        return w2idx, idx2w

    def tokenize(self, X, mapper):
        tokens = []
        for words in X:
            _tokens = []
            for word in words:
                assert mapper[word] is not None
                _tokens.append(mapper[word])
            tokens.append(_tokens)
        return tokens

    def untokenize(self, X, mapper):
        idx2w = self.idx2w
        words = []
        for tokens in X:
            _words = []
            for token in tokens:
                assert mapper[token] is not None
                _words.append(mapper[token])
            words.append(_words)
        return words

    def pad_tokens(self, tokens, pad_value, batch_first=False):
        padded_tokens = []
        for _tokens in tokens:
            _tokens = torch.tensor(_tokens)
            _tokens = _tokens.reshape(*_tokens.size()[::-1])
            padded_tokens.append(_tokens)
        padded_tokens = pad_sequence(padded_tokens, batch_first=batch_first, padding_value=pad_value)
        return padded_tokens

    def get_mask(self, tokens, pad_value):
        return torch.where(tokens != pad_value, True, False)

    def get_vocab_size(self, tokens):
        flattened_tokens = torch.unique(tokens)
        return flattened_tokens.size()[0]

    def add_beg_end_tag(self, tokens:[list], beg_value, end_value):
        modified_tokens = []
        for _tokens in tokens:
            _tokens = [beg_value] + _tokens + [end_value]
            modified_tokens.append(_tokens)
        return modified_tokens

if __name__ == '__main__':
    a = torch.ones(10, 3)
    # print(a.size()[::-1])
    print(a.resize(*a.size()[::-1]).size())
    # print(Helper.pad_tokens([a,b,c]).size())







