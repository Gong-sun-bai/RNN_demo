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

#选择网络模型
model_type ='LSTM'# # 选择'MLP'，'RNN'，'LSTM'或'GRU'

# 创建保存目录
log_dir = './logs'
image_dir = './images'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# 用于记录训练和验证损失
train_losses = []
val_losses = []

# 配置日志文件路径
log_dir = './logs'
log_file = os.path.join(log_dir, f'{model_type}_training.log')

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # 保存到文件
        logging.StreamHandler(sys.stdout)        # 同时输出到控制台
    ]
)

logger = logging.getLogger()

# 检查是否有 GPU
device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')


#定义超参数(注:这些超参数都可以自己调调看)
batch_size= 32
learning_rate = 0.001
num_epochs =20

#一、加载和预处理数据
data = pd.read_csv('jena_climate_2009_2016.csv')
selected_columns = data.columns[1:]  #选择所有特征列
data = data[selected_columns]
data = data.values.astype(np.float32) # 转换为numpy数组

# 分割数据集
num_train_samples = int(0.5 * len(data))
num_val_samples = int(0.25 * len(data))
num_test_samples = len(data) - num_train_samples - num_val_samples

train_data = data[:num_train_samples]
val_data = data[num_train_samples:num_train_samples + num_val_samples]
test_data = data[num_train_samples + num_val_samples:]

# 标准化数据（注意考虑实际预测这里只算训练集上的均值和方差）
mean = train_data.mean(axis=0) #pandas库操作，指定沿着列方向算
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
val_data = (val_data - mean) / std
test_data = (test_data - mean) / std

# 提取温度数据
temperature = data[:, 1]  # 温度在第二列
train_temperature = temperature[:num_train_samples]
val_temperature = temperature[num_train_samples:num_train_samples + num_val_samples]
test_temperature = temperature[num_train_samples + num_val_samples:]
mean_temp = train_temperature.mean()
std_temp = train_temperature.std()
train_temperature = (train_temperature - mean_temp) / std_temp
val_temperature = (val_temperature - mean_temp) / std_temp
test_temperature = (test_temperature - mean_temp) / std_temp

#创建输入输出对
def random_input(timeseries, length, target_time, target_data, sampling_rate = 1):
    (ts_length, dimensions) = timeseries.shape
    max_start = ts_length - length - target_time
    start = np.random.randint(0,max_start)
    end = start + length
    result_input = timeseries[start:end:sampling_rate, :]
    target = target_data[end+target_time-1]
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

seq_length = 5 * 24 * 6  # 5天的数据，每天144个数据
target_time = 24 * 6  # 24小时后的温度
sampling_rate = 6  # 每小时取一个样本
num_samples = 50000  # 生成的样本数量

train_inputs, train_targets = example_set(train_data, num_samples, seq_length, target_time, train_temperature, sampling_rate)
val_inputs, val_targets = example_set(val_data, num_samples, seq_length, target_time, val_temperature, sampling_rate)
test_inputs, test_targets = example_set(test_data, num_samples, seq_length, target_time, test_temperature, sampling_rate)

#转换为PyTorch张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_targets = torch.tensor(val_targets, dtype=torch.float32)
test_inputs = torch.tensor(test_inputs, dtype=torch.float32)
test_targets = torch.tensor(test_targets, dtype=torch.float32)

# 创建DataLoader
train_dataset = TensorDataset(train_inputs, train_targets) 
val_dataset = TensorDataset(val_inputs, val_targets)
test_dataset = TensorDataset(test_inputs, test_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#二、搭建网络模型
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.flatten =nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16,1)
    def forward(self, x):
        x= self.flatten(x)
        x= torch.tanh(self.fc1(x))#这里不能用relu，测试集损失会爆
        x= self.fc2(x)
        return x
    

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

#LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out  
      
#GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        
        return out

#################根据model_type选择模型##########################
if model_type == 'MLP':
    input_dim= 120*14
    model = MLP(input_dim).to(device)
elif model_type == 'LSTM':
    input_dim = 14  # 14个特征
    hidden_dim = 50
    output_dim = 1  # 预测单个温度值
    num_layers = 2
    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
elif model_type == 'RNN':
    input_dim = 14  # 14个特征
    hidden_dim = 50
    output_dim = 1  # 预测单个温度值
    num_layers = 2
    model = RNNModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
elif model_type == 'GRU':
    input_dim = 14  # 14个特征
    hidden_dim = 50
    output_dim = 1  # 预测单个温度值
    num_layers = 2
    model = GRUModel(input_dim, hidden_dim, output_dim, num_layers).to(device)
else:
    logger.info("Invaild model type.Choose MLP、LSTM、RNN or GRU ")
    raise ValueError("Invaild model type.Choose MLP、LSTM、RNN or GRU ")


#三、选择损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


print(f"The model is {model_type}")
logger.info(f"The model is {model_type}")

# 四、训练和验证
best_val_loss = float('inf')  # 初始化最佳验证损失
best_model_weights = None

for epoch in range(num_epochs):
    # 训练
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))  # 增加一个维度
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)  # 计算训练集平均损失
    train_losses.append(train_loss)  # 记录训练损失

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            val_loss += loss.item()
    val_loss /= len(val_loader)  # 计算验证集平均损失
    val_losses.append(val_loss)  # 记录验证损失

    # 打印训练和验证损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict().copy()

# 恢复最佳模型
model.load_state_dict(best_model_weights)

# 测试
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        test_loss += loss.item()
test_loss /= len(test_loader)  # 计算测试集平均损失
print(f'Test Loss: {test_loss:.4f}')
logger.info(f'Test Loss: {test_loss:.4f}')

# 保存损失数据到 .npy 文件
np.save(os.path.join(log_dir, f'{model_type}_train_losses.npy'), np.array(train_losses))
np.save(os.path.join(log_dir, f'{model_type}_val_losses.npy'), np.array(val_losses))

# 绘制损失曲线
plt.figure(figsize=(10, 6))
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o', color='blue')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o', color='orange')
plt.axhline(y=test_loss, color='red', linestyle='--', label='Test Loss')

# 在Test Loss线上标注损失值
plt.text(num_epochs, test_loss, f'{test_loss:.4f}', 
         color='red', fontsize=12, ha='right', va='bottom', 
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

# 设置图表标题和标签
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'{model_type} Training and Validation Loss')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig(os.path.join(image_dir, f'{model_type}_loss_curve.png'))
plt.show()
