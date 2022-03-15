import numpy as np
import torch
from torchcrf import CRF
from torch import nn
from tqdm import tqdm


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag2idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag2idx = tag2idx
        self.target_size = len(tag2idx)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(hidden_dim, self.target_size)
        self.crf = CRF(num_tags=self.target_size, batch_first=True)

    def forward(self, X, y, mask):
        X = self.embedding(X)
        o, (h, c) = self.lstm(X)
        X = self.linear(o)
        X = self.crf(X, y, mask=mask)
        return X

    def inference(self, X, mask):
        X = self.embedding(X)
        o, (h, c) = self.lstm(X)
        X = self.linear(o)
        X = self.crf.decode(emissions=X, mask=mask)
        return X

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def test(self, dataLoader, batch_size):
        model.eval()
        losses = []
        preds, truths = [], []
        for x, y, mask in tqdm(dataLoader):
            loss = -self.forward(x, y, mask) / batch_size
            losses.append(loss)
            pred = self.inference(x, mask)
            preds.extend(pred)
            for truth, pred in zip(y, pred):
                truths.append(truth[:len(pred)])
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        accuracy = np.where(preds == truths, 1, 0).mean()
        return accuracy


if __name__ == '__main__':
    tags = torch.ones(8)
    model = BiLSTM_CRF(128, tags, 128, 32)
    X = torch.randint(0, 128, (32, 4))
    mask = torch.ones(32, 4, dtype=torch.bool)
    y = torch.randint(0, 8, (32, 4))
    print(model)
    print(model.inference(X, mask))
    # model(X)

