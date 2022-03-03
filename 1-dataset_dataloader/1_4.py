# 1_4节：DataLoader实现方法

import random 
import numpy as np

class Dataset:
    def __init__(self, all_datas, batch_size, shuffle):
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        return DataLoader(self) 

    def __len__(self):
        return len(self.all_datas)
        

class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle:
            random.shuffle(self.index)
        self.cursor = 0

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        index = self.index[self.cursor:self.cursor+self.dataset.batch_size]
        batch_data = self.dataset.all_datas[index]
        self.cursor += self.dataset.batch_size 
        return batch_data


if __name__ == "__main__":
    all_datas = np.array([1, 2, 3, 4, 5, 6, 7])
    batch_size = 2
    epoch = 2 
    shuffle = True 

    dataset = Dataset(all_datas, batch_size, shuffle)
    for e in range(epoch):
        for data in dataset:
            print(data)