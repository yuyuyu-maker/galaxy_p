import torch.nn.functional as F
import torch
import torch.nn as nn

def swish(x):
    return x * torch.sigmoid(x)  # 或者直接使用 torch.sigmoid(x)

class GalaxyCNN3(nn.Module):
    def __init__(self, input_channels=3, num_classes=3):
        super(GalaxyCNN3, self).__init__()

        # 第一组卷积层 + 最大池化
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二组卷积层 + 最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三组卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 第四组卷积层（可选）
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 计算展平后的尺寸：256 * 16 * 16
        self.fc1 = nn.Linear(256 * 16 * 16, 512)  # 修正为 65536
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 第一组卷积
        x = F.relu(self.bn1(self.conv1(x)))
        # print(f"After conv1 + pool1: {x.size()}")
        x = self.pool1(x)
        x = self.dropout(x)

        # 第二组卷积
        x = swish(self.bn2(self.conv2(x)))
        # print(f"After conv2 + pool2: {x.size()}")
        x = self.pool2(x)
        x = self.dropout(x)

        # 第三组卷积
        x = F.relu(self.bn3(self.conv3(x)))
        # print(f"After conv3: {x.size()}")
        x = self.dropout(x)

        # 第四组卷积（如果添加了这一层）
        x = F.relu(self.bn4(self.conv4(x)))
        # print(f"After conv4: {x.size()}")
        x = self.dropout(x)

        # 展平多维的输入数据
        x = x.view(x.size(0), -1)  # 展平
        # print(f"After flatten: {x.size()}")  # 打印展平后的尺寸

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class GalaxyCNN2(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):  # 改为 1 作为二分类
        super(GalaxyCNN2, self).__init__()

        # 第一组卷积层 + 最大池化
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二组卷积层 + 最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三组卷积层
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 第四组卷积层（可选）
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # 计算展平后的尺寸：假设输入图像大小为 80x80
        self.fc1 = nn.Linear(256 * 20 * 20, 512)  #

        self.fc2 = nn.Linear(512, 1)  # 输出 1 作为二分类


        # Dropout层
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 第一组卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        # 第二组卷积
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # 第三组卷积
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        # 第四组卷积（可选）
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # 这里不需要手动加 sigmoid，BCEWithLogitsLoss 会处理
        return x


# 模型训练与评估