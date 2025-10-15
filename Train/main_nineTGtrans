import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import KFold
import time
from model_nineTGtrans import GODE
from matplotlib import pyplot as plt
from train import train, test
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
class Data(Dataset):
    def __init__(self, data,label):
        self.data=data
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx]

    def __len__(self):
        return len(self.data)

def onehot(inputs, num_classes):
    inputs = torch.tensor(inputs)  # 转换为 PyTorch 张量
    unique_labels = torch.unique(inputs)  # 获取唯一的类别值
    label_mapping = {label.item(): idx for idx, label in enumerate(unique_labels)}
    # 创建映射后的标签数组
    mapped_labels = torch.tensor([label_mapping[label.item()] for label in inputs])
    # 使用one_hot函数进行独热编码
    one_Y = torch.nn.functional.one_hot(mapped_labels, num_classes=num_classes)
    one_Y = one_Y.reshape((-1, num_classes))
    return one_Y

def normalize(data):
    for channel in range(data.shape[2]):
        mean = np.mean(data[:, :, channel])
        std = np.std(data[:, :, channel])
        data[:, :, channel] = (data[:, :, channel] - mean) / std
    return torch.tensor(data)

falx = np.load('./X_band.npy')
y = np.load('./y1_band.npy')
falx = falx[:, :, :, -1]
# 创建模型实例
num_classes = 3  # 根据你的问题设置类别数量
y = onehot(y, num_classes)
falx = normalize(falx)
# 将模型和数据移到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 训练模型
num_epochs = 50
batch_size = 64
mean_test_acc1 = []
mean_test_loss1 = []
std_test_acc1 = []
i = 0
milestones = [10]
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=40)
print("model begin")

for fold, (train_index, test_index) in enumerate(kf.split(falx)):
    model = GODE().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    train_losses=[]
    test_losses=[]
    test_accuracies=[]
    X_train, X_test = falx[train_index], falx[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    i = i + 1
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
  
        train_acc, train_loss = train(train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)
        # 模型评估
        test_acc, test_loss = test(test_loader, model, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        print('Epoch [{}/{}], 第 {} 组, Learning Rate:{}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}% '
            .format(epoch + 1, num_epochs, i, current_lr, train_loss, train_acc, test_loss, test_acc))
        scheduler.step()
    mean_test_acc1.append(test_accuracies[-1])
    mean_test_loss1.append(test_losses[-1])
    std_test_acc1.append(test_accuracies[-1])
print("Mean Test Accuracy: {:.2f}%".format(np.mean(mean_test_acc1)))
print("Mean Test Loss: {:.4f}".format(np.mean(mean_test_loss1)))
print("Std Test Accuracy: {:.2f}%".format(np.std(std_test_acc1)))


colors1 = ['r']
colors2 = ['purple', 'orange', 'c']
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(range(1, len(train_losses) + 1), train_losses, color=colors1[0],
        label=' Train Loss')

ax.set_title('Train Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(loc='upper right')
plt.savefig('train_loss_XTGODE_cross.png')
plt.close()

fig1, ax = plt.subplots(figsize=(8, 8))

ax.plot(range(1, len(test_losses) + 1), test_losses, color=colors1[0],
        label=' Test Loss')

ax.set_title('Test Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend(loc='upper right')
plt.savefig('test_los_XTGODE_transformer_cross.png')
plt.close()

    # Plotting test accuracies
fig2, ax = plt.subplots(figsize=(8, 8))

ax.plot(range(1, len(test_accuracies) + 1), test_accuracies, color=colors1[0],
            label=' Accuracy')

ax.set_title('Test Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.legend(loc='lower right')
plt.savefig('test_accuracy_XTGODE_transformer_cross.png')
plt.close()
