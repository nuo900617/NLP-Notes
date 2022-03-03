# 求解 (x-2)^2 = 0

from re import X
from tqdm import trange

x = 5.0
label = 0
epoch = 1000
lr = 0.01

for e in trange(epoch):
    pred = (x-2) ** 2 
    loss = (pred - label) ** 2
    deta_x = 4*(pred-label)*(x-2)
    x = x-deta_x*lr 

print(x)

