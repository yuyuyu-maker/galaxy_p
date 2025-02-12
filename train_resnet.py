import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# 评估模型并计算准确率
def evaluate(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取模型输出
            outputs = model(inputs)

            # 预测类别
            _, preds = torch.max(outputs, 1)

            # 如果标签是one-hot格式，转换为整数标签
            if labels.dim() > 1:  # one-hot编码
                _, labels = torch.max(labels, 1)

            # 保存预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

# 训练模型并绘制图像
def train_and_plot(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=15):
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
            optimizer.zero_grad()
            outputs = model(inputs)

            # 如果标签是one-hot格式，转换为整数标签
            if labels.dim() > 1:  # one-hot编码
                labels = torch.argmax(labels, dim=1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # 计算每个epoch的损失和准确率
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        test_accuracy = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Accuracy: {test_accuracy*100:.2f}%')

    # 绘制训练损失和准确率
    plt.figure(figsize=(12, 6))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练准确率与测试准确率
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
def evaluate_model(model, test_loader, device):
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
