import torch
from torchvision import transforms
from torchvision import datasets
from torchvision import models   # 加载模型
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import time

#from threading import Thread   # 创建线程绘制图像



#import numpy as np
import matplotlib.pyplot as plt   # 加载图像绘制库
from utils.utils import Utils   # 加载工具类

# 记录程序开始运行时间
start_time = time.process_time()

# 预定义参数
batch_size = 32   # 每批数据大小
epoch = 10

# transforms parameters
param_trans_RandomCrop, param_trans_RandomCrop_padding = 40, 4
param_trans_Resize = (224, 224)
param_trans_RandomHorizontalFlip = 0.5   # 随机水平翻转
param_trans_Normalize1, param_trans_Normalize2 = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # 归一化

# loss and optimizer parameters
learning_rate = 0.001   # 学习率 - 更新步长
param_optim_momentum = 0.9   # 越大越偏向历史梯度，越小越偏向当前梯度


# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(param_trans_RandomCrop, padding = param_trans_RandomCrop_padding),   # 随机裁剪
    transforms.Resize(param_trans_Resize),
    transforms.RandomHorizontalFlip(p = param_trans_RandomHorizontalFlip),
    transforms.ToTensor(),
    transforms.Normalize(param_trans_Normalize1, param_trans_Normalize2)])
""""""

# def init_data
# 1. 准备数据集
## 载入数据集和加载器 - 训练集
train_dataset = datasets.CIFAR10('./cifar-10-python', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)   # num_workers设置可能引起报错，在linux环境下可能正常

## 载入数据集和加载器 - 测试集
test_dataset = datasets.CIFAR10('./cifar-10-python', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


print('stage 1 successfully...')


# 2. 设计模型
#model = models.resnet18()   # 加载resnet18模型
#model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)   # 加载resnet18模型
"""
pretrained是旧版本写法
新版本为weights = models.ResNet101_Weights.DEFAULT
例如：models.resnet101(weights = models.ResNet101_Weights.DEFAULT)
"""
model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)   # 加载resnet50模型，参数pretrained可设置预训练参数
#model = models.resnet101()
inchannel = model.fc.in_features
model.fc = nn.Linear(inchannel, 10)   # 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层




print('stage 2 successfully...')

# 3. 构造损失函数和优化器
## GPU加速
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()   # 使用交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = param_optim_momentum)



print('stage 3 successfully...')



# 定义绘图数据
epoch_list = []
loss_list = []
accuracy_list = []


def draw_data(_x_label, _x_list, _y_label, _y_list, _z_label = '', acc_list = []):
    """
    用于绘制数据图像
    """
    plt.plot(_x_list, _y_list)
    if len(acc_list) != 0:
        plt.plot(_x_list, acc_list)
        for x, z in zip(_x_list, acc_list):
            plt.text(x, z + 0.01, '%.2f' % z, ha = 'center', va = 'bottom', fontsize = 12)
    plt.xlabel(_x_label)
    plt.ylabel(f'{_y_label} / {_z_label}')
    plt.title(Utils.get_title(batch_size=batch_size,
                epoch=epoch,
                RandomCrop=f'({param_trans_RandomCrop}, padding={param_trans_RandomCrop_padding})',
                Resize=param_trans_Resize,
                Normalize=(param_trans_Normalize1, param_trans_Normalize2),
                RandomHorizontalFlip=param_trans_RandomHorizontalFlip,
                lr=learning_rate,
                momentum=param_optim_momentum,
                ))
    plt.grid(True)
    # 标出数据点
    for x, y in zip(_x_list, _y_list):
        plt.text(x, y + 0.01, '%.2f' % y, ha = 'center', va = 'bottom', fontsize = 12)
    plt.show()



# 4. 定义训练循环
def train(epoch, step = 10):
    r"""
    Parameters
    ------------
    epoch : 训练轮数

    step : 迭代步长
    """

    for _epoch in range(epoch):
        _training_loss = 0.0   # 每次迭代损失
        _mean_loss = 0.0   # 每步迭代均损失

        for iteration, input_data in enumerate(train_loader):
            #inputs, target = input_data
            image, label = input_data[0].to(device), input_data[1].to(device)
            optimizer.zero_grad()   # 清空梯度

            # 正向传播
            outputs = model(image)
            loss = criterion(outputs, label)

            loss.backward()   # 反向传播

            optimizer.step()   # 更新参数

            _training_loss += loss.item()   # 计算当前损失

            # 每个迭代周期
            if iteration % step == 0:
                _mean_loss = _training_loss / step   # 计算每步迭代的均损失
                print(f'iteration - {_epoch * len(train_loader) + iteration} training loss: {_mean_loss}')
                _training_loss = 0.0
                print('总完成进度: %.2f%%  [epoch: %d/%d]  当前周期进度：%.2f%%' %
                      (((iteration + len(train_loader) * _epoch) / len(train_loader) * 100 / epoch),
                        (_epoch + 1),
                        (epoch), 
                        (iteration / len(train_loader) * 100)))
                print()


        # 输入绘图数据
        epoch_list.append(_epoch)
        loss_list.append(_mean_loss)
        test()   # 测试准确度
        #draw_data('Epoch', epoch_list, 'Loss', loss_list, 'Accuracy', accuracy_list)   # 测试


# 定义测试循环
def test():
    _correct = 0
    _total = 0
    with torch.no_grad():   # 测试集不需要计算梯度
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)   # 使用GPU加速
            outputs = model(images)  # 正向传播计算
            _, predicted = torch.max(outputs.data, dim=1)
            _total += labels.size(0)   # 取labels集的数据长度
            _correct += (predicted == labels).sum().item()
    accuracy = round(_correct / _total * 100, 2)
    print(f'Accuracy on test sets: {accuracy}%  [{_correct}/{_total}]')
    print()
    accuracy_list.append(accuracy)



train(epoch, 10)


print('stage 4 successfully...')

# 记录结束运行时间
end_time = time.process_time()

print('此次运行总耗时: ', round(end_time - start_time, 3), '秒')

draw_data('Epoch', epoch_list, 'Loss', loss_list, 'Accuracy', accuracy_list)   # 绘制数据图像




