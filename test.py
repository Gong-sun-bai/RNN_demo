import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
import os
import logging
import sys

os.chdir('/home/gdut_students/lwb/RNN_network')

# 超参数
batch_size = 128
num_epochs = 100
learning_rates = [0.1, 0.001, 0.000001]  # 三种学习率

# 日志和图片保存目录
log_dir = './logs'
image_dir = './images_learning_rate_experiment'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# 检查是否有 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
data = pd.read_csv('jena_climate_2009_2016.csv')
selected_columns = data.columns[1:]
data = data[selected_columns]
data = data.values.astype(np.float32)

# 分割数据集
num_train_samples = int(0.5 * len(data))
num_val_samples = int(0.25 * len(data))
num_test_samples = len(data) - num_train_samples - num_val_samples

train_data = data[:num_train_samples]
val_data = data[num_train_samples:num_train_samples + num_val_samples]
test_data = data[num_train_samples + num_val_samples:]

# 标准化数据
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

# 提取温度数据
temperature = data[:, 1]
train_temperature = temperature[:num_train_samples]
val_temperature = temperature[num_train_samples:num_train_samples + num_val_samples]
test_temperature = temperature[num_train_samples + num_val_samples:]
mean_temp = train_temperature.mean()
std_temp = train_temperature.std()
train_temperature = (train_temperature - mean_temp) / std_temp
val_temperature = (val_temperature - mean_temp) / std_temp
test_temperature = (test_temperature - mean_temp) / std_temp

# 创建输入输出对
def random_input(timeseries, length, target_time, target_data, sampling_rate=1):
    (ts_length, dimensions) = timeseries.shape
    max_start = ts_length - length - target_time
    start = np.random.randint(0, max_start)
    end = start + length
    result_input = timeseries[start:end:sampling_rate, :]
    target = target_data[end + target_time - 1]
    return (result_input, target)

def example_set(timeseries, number, length, target_time, target_data, sampling_rate=1):
    (ts_length, dimensions) = timeseries.shape
    input_length = math.ceil(length / sampling_rate)
    inputs = np.zeros((number, input_length, dimensions))
    targets = np.zeros((number))
    
    for i in range(number):
        (inp, target) = random_input(timeseries, length, target_time, target_data, sampling_rate)
        inputs[i] = inp
        targets[i] = target 

    return (inputs, targets)

seq_length = 5 * 24 * 6  # 5天的数据
target_time = 24 * 6  # 24小时后的温度
sampling_rate = 6  # 每小时取一个样本
num_samples = 50000

train_inputs, train_targets = example_set(train_data, num_samples, seq_length, target_time, train_temperature, sampling_rate)
val_inputs, val_targets = example_set(val_data, num_samples, seq_length, target_time, val_temperature, sampling_rate)
test_inputs, test_targets = example_set(test_data, num_samples, seq_length, target_time, test_temperature, sampling_rate)

train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_targets = torch.tensor(val_targets, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_targets = torch.tensor(test_targets, dtype=torch.float32)

train_dataset = TensorDataset(train_inputs, train_targets)
val_dataset = TensorDataset(val_inputs, val_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 开始实验
input_dim = 14
hidden_dim = 50
output_dim = 1
num_layers = 2

for lr in learning_rates:
    model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_weights = None

    log_file = os.path.join(log_dir, f'RNN_lr_{lr}_training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger()
    logger.info(f"Starting training for learning rate: {lr}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()

    model.load_state_dict(best_model_weights)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    logger.info(f"Test Loss for learning rate {lr}: {test_loss:.4f}")

    # 保存损失
    np.save(os.path.join(log_dir, f'RNN_lr_{lr}_train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(log_dir, f'RNN_lr_{lr}_val_losses.npy'), np.array(val_losses))

    # 绘图
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='orange')
    plt.axhline(y=test_loss, color='red', linestyle='--', label='Test Loss')
    plt.text(num_epochs, test_loss, f'{test_loss:.4f}', color='red', fontsize=12, ha='right', va='bottom', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'RNN Training and Validation Loss (LR={lr})')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(image_dir, f'RNN_lr_{lr}_loss_curve.png'))
    plt.show()
