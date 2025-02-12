import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
import random

import torch
from torchvision import transforms

train_transforms = transforms.Compose([
    # 随机裁剪至 240x240，并调整比例和大小
    transforms.RandomResizedCrop(240, scale=(0.7, 1.0), ratio=(0.8, 1.2)),

    # 数据增强
    transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
    transforms.RandomVerticalFlip(p=0.3),  # 30%概率垂直翻转
    transforms.RandomRotation(30),  # 随机旋转 0°~30°
    transforms.RandomAffine(
        degrees=15,  # 旋转角度范围
        translate=(0.1, 0.1),  # 水平和垂直平移幅度
        scale=(0.9, 1.1)  # 随机缩放
    ),

    # 颜色增强
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    # 随机透视变换
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # 30%概率

    # 高斯模糊
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),

    # 调整为目标尺寸 80x80
    transforms.Resize((80, 80)),

    # 转换为Tensor，并归一化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    # 将短边缩放到 256
    transforms.Resize(256),

    # 中心裁剪至 240x240
    transforms.CenterCrop(240),

    # 下采样至 80x80
    transforms.Resize((80, 80)),

    # 转换为Tensor，并归一化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 预处理函数


def preprocess_tensor(image_tensor, mode='train', device='cuda'):
    # 确保输入图像的形状正确
    if len(image_tensor.shape) != 3:
        raise ValueError("Input tensor must have shape (channels, height, width)")

    # 先检查输入的形状
    c, h, w = image_tensor.shape
    # print("Before transformation:", c, h, w)

    # 将图像转换为 PIL 图片格式
    pil_img = transforms.ToPILImage()(image_tensor)

    # 选择不同的 transform，处理图像
    processed_image = train_transforms(pil_img) if mode == 'train' else test_transforms(pil_img)

    # 确保返回的数据在指定设备上
    # print("After transformation:", processed_image.shape)
    return processed_image.to(device)


def random_sample(images_tensor, labels_tensor):
    # 随机选择一个索引
    idx = random.randint(0, len(images_tensor) - 1)

    # 获取图片和标签
    image = images_tensor[idx]
    label = labels_tensor[idx]

    # 将图像数据从 [0, 1] 转换为 [0, 255] 以便显示
    image = image.numpy() * 255
    image = image.transpose(1, 2, 0)  # 转换形状为 (128, 128, 3)
    print(f"图像编号: {idx}, 图像形状: {image.shape}")

    # 显示图像，并添加图片编号
    plt.imshow(image.astype(np.uint8))
    plt.title(f"编号: {idx}, 标签：SPIRAL={label[0]}, ELLIPTICAL={label[1]}, UNCERTAIN={label[2]}")
    plt.axis('off')  # 关闭坐标轴
    plt.show()

def print_dataset_info(images_tensor, labels_tensor):
    # 遍历数据集中的每一个数据
    for idx in range(len(images_tensor)):
        # 获取当前图像和标签
        image = images_tensor[idx]
        label = labels_tensor[idx]
        image = image.numpy() * 255
        image = image.transpose(1, 2, 0)  # 转换形状为 (128, 128, 3)
        plt.imshow(image.astype(np.uint8))
        plt.title(f"编号: {idx}, 标签：SPIRAL={label[0]}, ELLIPTICAL={label[1]}, UNCERTAIN={label[2]}")
        plt.axis('off')  # 关闭坐标轴
        plt.show()
        # 打印数据编号和标签
        print(f"数据编号: {idx}, 标签：SPIRAL={label[0]}, ELLIPTICAL={label[1]}, UNCERTAIN={label[2]}")
# 调用函数，随机检测一个样本

def print_dataset_info2(images_tensor, labels_tensor, spire_indices, elliptical_indices):
    spire_count, elliptical_count = 0, 0

    # 遍历所有图像
    for idx in range(len(images_tensor)):
        label = labels_tensor[idx]
        if label[0] == 1:
            spire_count += 1
            spire_indices.append(idx)
        if label[1] == 1:
            elliptical_count += 1
            elliptical_indices.append(idx)
        # if label[2] == 1:
            # uncertain_count += 1
            # uncertain_indices.append(idx)

    # 输出每类的数量
    print(
        f"SPIRE星系一共有：{spire_count}个，ELLIPTICAL星系一共有{elliptical_count}个")

    return spire_indices, elliptical_indices, spire_count, elliptical_count


def Stratified_Sampling(images_tensor, labels_tensor, spire_indices, elliptical_indices, uncertain_indices, spire_count,
                        elliptical_count, uncertain_count):
    # 从SPIRE类别中抽样
    spire_sample_indices = np.random.choice(spire_indices, size=spire_count // 2, replace=False)
    spire_images = images_tensor[spire_sample_indices]
    spire_labels = labels_tensor[spire_sample_indices]

    # 从ELLIPTICAL类别中抽样
    elliptical_sample_indices = np.random.choice(elliptical_indices, size=elliptical_count // 2, replace=False)
    elliptical_images = images_tensor[elliptical_sample_indices]
    elliptical_labels = labels_tensor[elliptical_sample_indices]

    # 从UNCERTAIN类别中抽样
    uncertain_sample_indices = np.random.choice(uncertain_indices, size=uncertain_count // 2, replace=False)
    uncertain_images = images_tensor[uncertain_sample_indices]
    uncertain_labels = labels_tensor[uncertain_sample_indices]

    # 拼接所有抽样的图像和标签
    all_images = torch.cat([spire_images, elliptical_images, uncertain_images], dim=0)
    all_labels = torch.cat([spire_labels, elliptical_labels, uncertain_labels], dim=0)

    # 打乱所有的图像和标签
    shuffle_indices = np.random.permutation(len(all_images))
    all_images = all_images[shuffle_indices]
    all_labels = all_labels[shuffle_indices]

    return all_images, all_labels
