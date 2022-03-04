# 矩阵实现线性回归

# pandas skiprows=1：从第1行开始读
# pandas names：定义列名

import random
import numpy as np 
import pandas as pd 
from tqdm import trange


def load_data(file="上海二手房价.csv"):
    data = pd.read_csv(file, names=['y', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'], skiprows=1)
    Y = data['y'].values.reshape(-1,1)
    X = data[[f"x{i}" for i in range(1, 7)]].values
    mean_x, std_x = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
    mean_y, std_y = np.mean(Y), np.std(Y)
    X = (X - mean_x) / std_x
    Y = (Y - mean_y) / std_y 
    return X, Y, mean_x, std_x, mean_y, std_y 


class Dataset:
    def __init__(self, X, Y, batch_size, shuffle=True):
        self.X = X
        self.Y = Y 
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        return DataLoader(self) 

    def __len__(self):
        return self.X.shape[0]


class DataLoader:
    def __init__(self, Dataset):
        self.Dataset = Dataset 
        self.cursor = 0
        self.index = [i for i in range(len(self.Dataset))]
        if self.Dataset.shuffle:
            random.shuffle(self.index)

    def __next__(self):
        if self.cursor >= len(self.Dataset):
            raise StopIteration
        index = self.index[self.cursor:self.cursor+self.Dataset.batch_size]
        batch_x = self.Dataset.X[index]
        batch_y = self.Dataset.Y[index]
        self.cursor += self.Dataset.batch_size
        return batch_x, batch_y


def main():
    batch_size = 279
    shuffle = True 
    X, Y, mean_x, std_x, mean_y, std_y = load_data(file="上海二手房价.csv")
    dataset = Dataset(X, Y, batch_size, shuffle)

    K = np.random.rand(X.shape[1], 1)
    b = np.random.rand(1)
    lr = 0.001
    epoch = 20000
    for e in trange(epoch):
        for x, y in dataset:
            pred = x @ K + b
            loss = np.sum((pred - y)**2)/batch_size
            G = 2*(pred-y)/batch_size
            deta_K = x.T @ G 
            deta_b = np.mean(G @ y.T)
            K -= deta_K * lr 
            b -= deta_b * lr
        if e % 1000 == 0:
            print(f"loss:{loss:.3f}")

    bedroom = 1 #(int(input("请输入卧室数量:")))
    ting = 1 #(int(input("请输入客厅数量:")))
    wei =  1 #(int(input("请输入卫生间数量:")))
    area = 41.13 # (int(input("请输入面积:")))
    floor = 6 # (int(input("请输入楼层:")))
    year = 1993 #(int(input("请输入建成年份:")))

    test_x = (np.array([bedroom, ting, wei, area, floor, year]).reshape(1, -1) - mean_x) / std_x

    p = test_x @ K + b
    print("房价为: ", p * std_y + mean_y)




if __name__ == "__main__":
    main()





