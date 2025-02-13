import pandas as pd
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import data_ycl
from torch.utils.data import DataLoader, Dataset
import train_cnn
from imblearn.over_sampling import SMOTE
from torch.utils.data.sampler import WeightedRandomSampler
import module

rcParams['font.family'] = 'SimHei'
# 设置路径
dataset_path = "C:/Users/panli/Desktop/galaxy zoo/dataset"
images_path = "C:/Users/panli/Desktop/galaxy zoo/images"

# 读取分类标签文件
class_file = os.path.join(dataset_path, "training_solutions_rev1.csv")
class_data = pd.read_csv(class_file)

# 读取星系编号和图片编号的映射文件
image_file = os.path.join(dataset_path, "image_ids.csv")  # 假设图片编号文件是这个名字
image_data = pd.read_csv(image_file)

# 合并两个数据集，通过 'GalaxyID' 进行连接
merged_data = pd.merge(class_data, image_data, on='OBJID')
merged_data = merged_data.drop('OBJID', axis=1)

# 现在 merged_data 包含了每个 GalaxyID 和对应的分类标签以及图片编号
# 你可以通过图片编号加载图片，并且对应它们的分类标签

def load_image(asset_id, resize_to=(64, 64)):
    """
    根据图片ID加载图片
    假设图片文件是JPEG格式，且图片编号与文件名相关
    如果图片大小不一致，resize_to 用来调整图片的大小
    """
    image_path = os.path.join(images_path, f"{asset_id}.jpg")
    if not os.path.exists(image_path):  # 如果图片文件不存在，跳过该图片
        # print(f"图片 {image_path} 不存在，跳过...")
        return None
    image = Image.open(image_path)
    image = image.resize(resize_to)  # 调整图片大小
    image = np.array(image) / 255.0  # 归一化处理，像素值变为 [0, 1]

    # 转换为 [channels, height, width] 格式
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 调整维度顺序
    print(f"图片 {image_path} 存在")
    return image

# 创建空列表，用于存储数据（图片和标签）
images = []
labels = []
spire=0
elliptical=0
# uncertain=0
spire_indices=[]
elliptical_indices=[]
# uncertain_indices=[]

class CustomDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, mode='train', device='cuda'):
        self.images_tensor = images_tensor
        self.labels_tensor = labels_tensor
        self.mode = mode
        self.device = device

    def __len__(self):
        # 返回数据集的长度
        return len(self.images_tensor)

    def __getitem__(self, idx):
        # 获取单张图片和标签
        image = self.images_tensor[idx]
        label = self.labels_tensor[idx]

        # 确保每次处理的是单张图片
        image = data_ycl.preprocess_tensor(image, mode=self.mode, device=self.device)

        # 将处理后的图像和标签返回，确保它们在目标设备上
        image = image.to(self.device)  # 将图像移动到指定设备
        label = label.to(self.device)  # 将标签移动到指定设备

        return image, label

# 模型训练与评估

if __name__ == "__main__":
    # 创建空列表，用于存储数据（图片和标签）
    images = []
    labels = []

    for idx, row in merged_data.iterrows():
        image_id = row['asset_id']  # 获取图片编号
        spiral = row['SPIRAL']
        elliptical = row['ELLIPTICAL']

        # One-hot 编码
        label = torch.tensor([elliptical, spiral], dtype=torch.float32)  # 形状 [2]

        image = load_image(image_id)
        if image is None:
            continue

        if elliptical in [0, 1] and spiral in [0, 1] and not (elliptical == 0 and spiral == 0):

            images.append(image)
            labels.append(label)
        else:
            print(f"跳过第 {idx} 个样本，标签值异常: ELLIPTICAL={elliptical}, SPIRAL={spiral}")

    # 确保数据不为空
    if len(images) == 0:
        print("没有加载任何图片，检查是否所有图片文件都丢失")
    else:
        images_tensor = torch.stack(images)
        labels_tensor = torch.stack(labels)  # 形状 [batch_size, 2]

        print("标签分布：", torch.unique(labels_tensor, return_counts=True, dim=0))

        torch.save(images_tensor, 'images.pt')
        torch.save(labels_tensor, 'labels.pt')

        print("数据已保存为张量")

    # 计算每个类的权重
    # 确保 labels 是张量类型]
    # print(labels)  # 查看 labels 的数据结构
    # 先划分数据集
    # 先划分数据集
    train_images, test_images, train_labels, test_labels = train_test_split(
        images_tensor, labels_tensor, test_size=0.2, random_state=42
    )

    # 通过 argmax 将 one-hot 转换为单标签
    train_labels = train_labels.argmax(dim=1).to(torch.int64)  # [N]
    test_labels = test_labels.argmax(dim=1).to(torch.int64)  # [N]

    # 重新计算 `train_labels` 的类别权重
    class_counts = torch.bincount(train_labels)
    class_weights = 1. / (class_counts.float() + 1e-6)  # 防止除零
    train_sample_weights = class_weights[train_labels]

    # 仅对训练集使用 `WeightedRandomSampler`
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_labels),  # 确保采样数量与 `train_labels` 匹配
        replacement=True
    )

    # 创建数据集
    train_dataset = CustomDataset(train_images, train_labels, mode='train', device='cuda')
    test_dataset = CustomDataset(test_images, test_labels, mode='test', device='cuda')

    # 训练集使用 `sampler`，测试集 `shuffle=False`
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

    # 在训练循环中打印每个批次的输入和输出形状

    # train_image = data_ycl.preprocess_tensor(train_loader, mode='train')
    # test_image = data_ycl.preprocess_tensor(test_loader, mode='test')

    # train_dataset = TensorDataset(train_images, train_labels)
    # test_dataset = TensorDataset(test_images, test_labels)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = module.GalaxyCNN2().to(device)
    # inputs, labels = inputs.to(device), labels.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    class_counts = torch.tensor([spire, elliptical], dtype=torch.float32)
    pos_weight = class_counts[0] / class_counts[1]  # 计算类别权重
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)



    train_cnn.train_and_plot2(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=5)
    cm = train_cnn.evaluate_model2(model, test_loader, device)
    # print("Confusion Matrix:")
    # print(cm)
    # 保存模型
    torch.save(model.state_dict(), "galaxy_cnn.pth")
    print("模型已保存为 'galaxy_cnn.pth'")


