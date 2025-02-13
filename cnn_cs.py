import torch
from PIL import Image
from matplotlib import rcParams
import pandas as pd
import random
import module
from torch.utils.data import DataLoader
import os
import data_ycl
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

def predict_image_tensor(images_tensor):
    random_idx = random.randint(0, len(images_tensor) - 1)
    img_tensor = images_tensor[random_idx]
    img_true_label = labels_tensor[random_idx]
    # 确保图像张量也在正确的设备上
    img_tensor = img_tensor.to(device)  # 将图像数据移到同一设备（GPU 或 CPU）
    img_tensor = data_ycl.preprocess_tensor(img_tensor, mode='test', device=device)
    img_tensor = img_tensor.unsqueeze(0)

    # 将图像送入模型
    with torch.no_grad():  # 不计算梯度
        output = model(img_tensor)

    # 获取预测的类别
    _, predicted_class = torch.max(output, 1)

    # 假设模型输出的是二分类（0或1）
    if predicted_class.item() == 1:
        prediction = "elliptical"
    else:
        prediction = "spiral"

    if img_true_label.argmax(dim=0).item() == 0:
        # 如果 one-hot 编码的真实标签是 [1, 0]（即标签为 0）
        true_label = "elliptical"
    elif img_true_label.argmax(dim=0).item() == 1:
        # 如果 one-hot 编码的真实标签是 [0, 1]（即标签为 1）
        true_label = "spiral"

    return prediction, true_label


# 预测函数
def predict_image(model, img_path):
    # 加载图像
    img = Image.open(img_path).convert("RGB")

    # 定义转换过程：将图像转换为张量
    transform = transforms.ToTensor()  # 只转换为张量，不做标准化

    # 将图像转换为张量
    img_tensor = transform(img).to(device)
    # 预处理图像
    img_tensor = data_ycl.preprocess_tensor(img_tensor, mode='test', device='cuda')
    img_tensor = img_tensor.unsqueeze(0)  # 扩展为batch维度 (1, C, H, W)

    # 将图像送入模型
    with torch.no_grad():  # 不计算梯度
        output = model(img_tensor)

    # 获取预测的类别
    _, predicted_class = torch.max(output, 1)

    # 假设模型输出的是二分类（0或1）
    if predicted_class.item() == 0:
        return "spiral"
    else:
        return "elliptical"


def load_model(model_path):
    model = module.GalaxyCNN2()  # 用你实际的模型类来替换
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
    return model


def predict(img_path, model_path="your_model.pth"):
    model = load_model(model_path)
    prediction = predict_image(model, img_path)
    return prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 确保在主程序中运行
if __name__ == "__main__":

    # 读取保存的张量
    images_tensor = torch.load('images.pt')
    labels_tensor = torch.load('labels.pt')

    # 打印数据的形状
    print(f"Images tensor shape: {images_tensor.shape}")
    print(f"Labels tensor shape: {labels_tensor.shape}")

    model = module.GalaxyCNN2().to(device)  # 将模型移到设备
    model.load_state_dict(torch.load("galaxy_cnn.pth"))
    dataset = TensorDataset(images_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)



    # 假设你的图像张量已经被加载
    images_tensor = torch.load('images.pt').to(device)  # 将图像数据移到同一设备（GPU 或 CPU）

    # 调用预测函数
    prediction , true_label= predict_image_tensor(images_tensor)
    print(f"Predicted class: {prediction}",f"Label: {true_label}")
