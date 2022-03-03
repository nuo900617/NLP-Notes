# 1_2节：基本功能实现

import random 

list1 = [1, 2, 3, 4, 5, 6, 7]

batch_size = 2
epoch = 2

for e in range(epoch):
    random.shuffle(list1)
    for i in range(0,len(list1),batch_size):
        batch_data = list1[i:i+batch_size]
        print(batch_data)