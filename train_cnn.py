from matplotlib import pyplot as plt
import torch
# from scipy.interpolate._ppoly import evaluate
from sklearn.metrics import accuracy_score
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn


def evaluate3(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 将标签调整为 (16, 1) 形状
            labels = labels.unsqueeze(1)  # 在第1维增加一个维度

            # 获取模型输出
            outputs = model(inputs)

            # 预测类别
            _, preds = torch.max(outputs, 1)

            # 获取真实标签的类别（从 one-hot 转换为整数标签）
            _, labels = torch.max(labels, 1)

            # 保存预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def train_and_plot3(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=15):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 将标签调整为 (16, 1) 形状
            labels = labels.unsqueeze(1)  # 在第1维增加一个维度

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == torch.argmax(labels, dim=1)).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        test_accuracy = evaluate3(model, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%')

    # 绘制训练损失和准确率
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# 评估模型并计算混淆矩阵
def evaluate_model3(model, test_loader, device):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型预测
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 获取预测的类别索引（最大值索引）

            # 如果标签是one-hot格式，转换为整数标签
            if labels.dim() > 1:
                _, labels = torch.max(labels, 1)

            all_preds.extend(preds.cpu().numpy())  # 将预测结果存入列表
            all_labels.extend(labels.cpu().numpy())  # 将真实标签存入列表

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    return cm


def evaluate2(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型输出 [batch_size, 1]
            outputs = model(inputs)

            # 将 preds 的形状从 [batch_size, 1] 转换为 [batch_size]
            preds = (torch.sigmoid(outputs) > 0.5).long().squeeze(1)  # .squeeze(1) 去掉多余的维度

            # 如果 labels 是 [batch_size, 1]，去掉第二维
            if labels.dim() == 2 and labels.shape[1] == 1:
                labels = labels.squeeze(1)  # 变成 [batch_size]

            # 确保 predictions 和 labels 的形状一致
            assert preds.shape == labels.shape, f"Shape mismatch: preds {preds.shape}, labels {labels.shape}"

            all_labels.extend(labels.cpu().numpy())  # 添加真实标签
            all_preds.extend(preds.cpu().numpy())    # 添加预测结果

            correct += (preds == labels).sum().item()  # 计算正确预测的个数
            total += labels.size(0)  # 计算总样本数

    accuracy = correct / total  # 准确率
    return accuracy



def train_and_plot2(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=20):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 确保 labels 是正确的形状（BCE 需要 (batch_size, 1)）
            if labels.dim() > 1 and labels.shape[1] == 2:
                labels = labels[:, 1]  # 只取正类（例如 ELLIPTICAL）的概率
            labels = labels.float().unsqueeze(1)  # 确保是 [batch_size, 1]

            optimizer.zero_grad()
            outputs = model(inputs)  # outputs 形状: [batch_size, 1]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 预测类别（BCEWithLogitsLoss 需要手动 sigmoid）
            preds = (torch.sigmoid(outputs) > 0.5).long()

            # 计算准确率
            correct += (preds == labels.round()).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        test_accuracy = evaluate2(model, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print(f"Correct: {correct}, Total: {total}")
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, '
              f'Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%')

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# **二分类任务应使用 BCEWithLogitsLoss**
criterion = nn.BCEWithLogitsLoss()



def evaluate_model2(model, test_loader, device):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型输出
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # 获取预测的类别索引（最大值索引）

            all_preds.extend(preds.cpu().numpy())  # 将预测结果存入列表
            all_labels.extend(labels.cpu().numpy())  # 将真实标签存入列表

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    return cm