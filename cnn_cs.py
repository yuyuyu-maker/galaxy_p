import torch
from matplotlib import rcParams
import galaxycnn
import pandas as pd
import random
from torch.utils.data import DataLoader
import os
from torch.utils.data import TensorDataset, DataLoader

# 读取保存的张量
images_tensor = torch.load('images.pt')
labels_tensor = torch.load('labels.pt')

# 打印数据的形状
print(f"Images tensor shape: {images_tensor.shape}")
print(f"Labels tensor shape: {labels_tensor.shape}")



# 创建TensorDataset
dataset = TensorDataset(images_tensor, labels_tensor)

# 创建DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# 遍历DataLoader中的数据
for batch_idx, (img_batch, label_batch) in enumerate(loader):
    # 这里可以进行模型的推理等操作
    print(img_batch.shape, label_batch.shape)
