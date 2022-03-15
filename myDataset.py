import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence
import helper as my_helper
import numpy as np

class myDataset(Dataset):
    def __init__(self, src_path, tgt_path, limit=None, encoding='utf-8'):
        super(myDataset, self).__init__()
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.encoding = encoding
        self.limit = limit
        # 读入的原始数据
        source = self.load_data(self.src_path, self.encoding)
        target = self.load_data(self.tgt_path, self.encoding)
        self.helper = my_helper.Helper(source, target)
        # tokenize过的X和y
        self.X = self.helper.tokenize(source, self.helper.w2idx)
        self.y = self.helper.tokenize(target, self.helper.tag2idx)

        del source, target
        # tokens => ['<BEG>'] + tokens + ['<END>']
        self.X = self.helper.add_beg_end_tag(self.X, self.helper.w2idx['<BEG>'], self.helper.w2idx['<END>'])
        self.y = self.helper.add_beg_end_tag(self.y, self.helper.tag2idx['<BEG>'], self.helper.tag2idx['<END>'])
        # pad 将所有句子补全为相同长度
        tmpX, tmpy = self.X,self.y
        self.X = self.helper.pad_tokens(self.X, batch_first=True, pad_value=self.helper.w2idx['<PAD>'])
        self.y = self.helper.pad_tokens(self.y, batch_first=True, pad_value=self.helper.tag2idx['<PAD>'])
        del tmpX, tmpy
        # 获取mask => [1, 1, 1, 1, 1, 0, 0] 有效 = 1 无效 = 0
        self.mask = self.helper.get_mask(self.X, self.helper.w2idx['<PAD>'])
        # 获取词汇数量 标签数量
        self.vocab_size = self.helper.get_vocab_size(self.X)
        self.tag_size = self.helper.get_vocab_size(self.y)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        mask = self.mask[idx]
        return X, y, mask

    def load_data(self, path, encoding):
        data = []
        with open(path, encoding=encoding) as f:
            for idx, line in enumerate(f.readlines()):
                if self.limit and idx >= self.limit:
                    break
                data.append(line.split())
        return data


if __name__ == '__main__':
    src_path = './data/source_BIO_2014_cropus.txt'
    tgt_path = './data/target_BIO_2014_cropus.txt'
    dataset = myDataset(src_path, tgt_path)
    print(dataset[0])

