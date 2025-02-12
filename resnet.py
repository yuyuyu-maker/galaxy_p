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
import torchvision.models as models
import train_resnet


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
    # print(f"图片 {image_path} 存在")
    return image


# 创建空列表，用于存储数据（图片和标签）
images = []
labels = []
flag=0
# 对每个星系进行处理
for idx, row in merged_data.iterrows():
    image_id = row['asset_id']  # 获取图片的唯一编号
    # 假设 SPIRAL、ELLIPTICAL 和 UNCERTAIN 列分别是类标签
    spiral = row['SPIRAL']
    elliptical = row['ELLIPTICAL']
    uncertain = row['UNCERTAIN']

    # 创建 one-hot 编码标签
    label = torch.tensor([spiral, elliptical, uncertain], dtype=torch.float32)
    flag+=1
    # 加载图片
    image = load_image(image_id)

    if image is None:
        continue
        # 将图片和标签分别添加到列表
    if (flag%3==0 and label != [0,0,1]):
        images.append(image)
        labels.append(label)


if len(images) == 0:
    print("没有加载任何图片，检查是否所有图片文件都丢失")
else:
    # 将图片和标签转换为张量形式（可以进一步处理）
    images_tensor = torch.stack(images)
    labels_tensor = torch.stack(labels)

    # 现在，你有了 images_tensor 和 labels_tensor，可以用来训练模型
    torch.save(images_tensor, 'images.pt')
    torch.save(labels_tensor, 'labels.pt')

    print("数据已保存为张量")

spireelliptical=0
# uncertain=0
spire_indices=[]
elliptical_indices=[]
# uncertain_indices=[]

# data_ycl.random_sample(images_tensor, labels_tensor)
# data_ycl.print_dataset_info(images_tensor, labels_tensor)
# spire_indices, elliptical_indices, uncertain_indices, spire, elliptical, uncertain = data_ycl.print_dataset_info2(images_tensor, labels_tensor, spire_indices, elliptical_indices, uncertain_indices)
# images_tensor,labels_tensor=data_ycl.Stratified_Sampling(images_tensor, labels_tensor, spire_indices, elliptical_indices, uncertain_indices, spire,
                        # elliptical, uncertain)
# 随机选择一半的数据

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

# 数据集划分
train_images, test_images, train_labels, test_labels = train_test_split(images_tensor, labels_tensor, test_size=0.2, random_state=42)

# 创建 CustomDataset 实例
train_dataset = CustomDataset(train_images, train_labels, mode='train', device='cuda')
test_dataset = CustomDataset(test_images, test_labels, mode='test', device='cuda')

# 创建 DataLoader 实例
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)

# 在训练循环中打印每个批次的输入和输出形状

# train_image = data_ycl.preprocess_tensor(train_loader, mode='train')
# test_image = data_ycl.preprocess_tensor(test_loader, mode='test')

# train_dataset = TensorDataset(train_images, train_labels)
# test_dataset = TensorDataset(test_images, test_labels)

class ResNet3Class(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet3Class, self).__init__()

        # 加载预训练的 ResNet-18 模型
        self.resnet = models.resnet18(pretrained=True)

        # 修改全连接层（fc层）以适应三分类任务
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        # 如果需要，可以冻结卷积层的参数，防止其在训练过程中更新
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        # 解冻最后一层的参数，便于训练
        # for param in self.resnet.fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


if __name__ == "__main__":

    model = ResNet3Class(num_classes=3)

    # 打印模型结构
    print(model)

    # 创建一个虚拟输入张量进行测试
    input_tensor = torch.randn(16, 3, 64, 64)  # batch size 为 1，RGB 图像，80x80 尺寸
    output = model(input_tensor)

    print(f"Output shape: {output.shape}")  # 输出形状应该是 (1, 3)
# 模型训练与评估


# 创建模型实例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型并移动到 GPU（如果可用）
model = ResNet3Class(num_classes=3)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 适用于多类分类任务
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练并绘制图像
train_resnet.train_and_plot(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=15)

# 计算混淆矩阵
cm = train_resnet.evaluate_model(model, test_loader, device)
print("Confusion Matrix:")
print(cm)


# 保存模型
torch.save(model.state_dict(), "resnet.pth")
print("模型已保存为 'resnet.pth'")
