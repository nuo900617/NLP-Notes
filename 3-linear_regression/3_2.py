# Dataset & DataLoader

import numpy as np
import random
from sklearn.utils import shuffle
from tqdm import trange

class Dataset:
    def __init__(self, datas, labels, batch_size, shuffle=True):
        self.datas = datas 
        self.labels = labels
        self.batch_size = batch_size 
        self.shuffle = shuffle

    def __iter__(self):
        return DataLoader(self)

    def __len__(self):
        return len(self.datas)


class DataLoader:
    def __init__(self, dataset) -> None:
        self.dataset = dataset 
        self.cursor = 0
        self.index = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle:
            random.shuffle(self.index)

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        index = self.index[self.cursor:self.cursor+self.dataset.batch_size]
        batch_data = self.dataset.datas[index]
        batch_label = self.dataset.labels[index]
        self.cursor += self.dataset.batch_size
        return batch_data, batch_label 


if __name__ == "__main__":

    years = np.array([i for i in range(2000, 2022)])
    prices = np.array([10000,11000,12000,13000,14000,12000,13000,16000,18000,20000,19000,22000,24000,23000,26000,35000,30000,40000,45000,52000,50000,60000])
    years_min = np.min(years)
    years_max = np.max(years)
    years = (years - years_min)/(years_max - years_min)
    prices_min = np.min(prices)
    prices_max = np.max(prices)
    prices = (prices - prices_min)/(prices_max - prices_min)

    epoch = 10000
    batch_size = 10
    shuffle = True

    k = 2
    b = 0
    lr = 0.05

    dataset = Dataset(years, prices, batch_size)
    for e in trange(epoch):
        for data, label in dataset:
            pred = k * data + b 
            loss = (pred - label) ** 2

            deta_k = 2 * (pred - label) * data 
            deta_b = 2 * (pred - label)

            k -= np.sum(deta_k)/batch_size * lr 
            b -= np.sum(deta_b)/batch_size * lr 

    print('k = {}, b = {}'.format(k, b))

    year = 2022
    price = k * (year-years_min)/(years_max-years_min) + b
    price = price * (prices_max - prices_min) + prices_min
    print("price: ", price)
