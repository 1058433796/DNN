import numpy as np

from myModel import *
from myDataset import *
from tqdm import tqdm, trange
from torch.utils.data import random_split
import time
from myPlot import MyPlot

src_path = './data/source_BIO_2014_cropus.txt'
tgt_path = './data/target_BIO_2014_cropus.txt'

epoches = 10
test_ratio = 0.2
batch_size = 64
embedding_dim = 128
hidden_dim = 32
save_path = './checkpoint/myModel.ckpt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('current device:', device)

if __name__ == '__main__':
    dataset = myDataset(src_path, tgt_path, limit=1024)
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset =\
        random_split(
            dataset,(train_size, test_size)
        )
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataLoader = DataLoader(test_dataset, batch_size=batch_size)
    model = BiLSTM_CRF(dataset.vocab_size, dataset.helper.tag2idx, embedding_dim, hidden_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    train_losses, test_losses = [], []
    test_preds, test_truths = [], []

    for epoch in range(epoches):
        for x, y, mask in tqdm(train_dataLoader):
            loss = -model.forward(x, y, mask) / batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())
    t = time.localtime()
    month, day, hour, minute = t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min
    model.save_model(save_path + f'{month}-{day} {hour}:{minute}')
    accuracy = model.test(test_dataLoader, batch_size)
    MyPlot.plot_curve(range(epoches), train_losses, l='train_loss')
    MyPlot.plot_curve(range(len(test_losses)), test_losses, l='test_losses')
    print('the accuracy of test is', accuracy)











