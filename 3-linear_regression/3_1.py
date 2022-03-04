# 基础功能实现
# y = kx + b

import numpy as np
from tqdm import trange

years = np.array([i for i in range(2000, 2022)])
prices = np.array([10000,11000,12000,13000,14000,12000,13000,16000,18000,20000,19000,22000,24000,23000,26000,35000,30000,40000,45000,52000,50000,60000])

prices = (prices - np.min(prices))/(np.max(prices) - np.min(prices))
print(prices)

k = 1
b = 0
epoch = 100
lr = 0.01

for e in trange(epoch):
    for x, label in zip(prices, years):
        pred = k*x + b
        loss = (pred - label) ** 2
        deta_k = 2*(pred-label)*x 
        deta_b = 2*(pred-label)

        k = k - deta_k * lr 
        b = b - deta_b * lr

print('k: ', k, ' b: ', b)
print((k*2023+b)*(np.max(prices)-np.min(prices))+np.min(prices))
