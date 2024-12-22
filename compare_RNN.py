import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt

# 通用的 RNN 模型定义
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

# 数据加载与预处理函数
def load_data(logger):
    logger.info("Loading and preprocessing data...")
    print("Loading and preprocessing data...")
    data = pd.read_csv('jena_climate_2009_2016.csv')
    selected_columns = data.columns[1:]
    data = data[selected_columns].values.astype(np.float32)
    
    num_train_samples = int(0.5 * len(data))
    num_val_samples = int(0.25 * len(data))
    train_data = data[:num_train_samples]
    val_data = data[num_train_samples:num_train_samples + num_val_samples]
    test_data = data[num_train_samples + num_val_samples:]
    
    mean, std = train_data.mean(axis=0), train_data.std(axis=0)
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std
    test_data = (test_data - mean) / std

    temperature = data[:, 1]
    mean_temp, std_temp = temperature[:num_train_samples].mean(), temperature[:num_train_samples].std()
    train_temperature = (temperature[:num_train_samples] - mean_temp) / std_temp
    val_temperature = (temperature[num_train_samples:num_train_samples + num_val_samples] - mean_temp) / std_temp
    test_temperature = (temperature[num_train_samples + num_val_samples:] - mean_temp) / std_temp

    seq_length = 5 * 24 * 6
    target_time = 24 * 6
    sampling_rate = 6
    num_samples = 50000

    def random_input(ts, length, target_time, target):
        max_start = len(ts) - length - target_time
        start = np.random.randint(0, max_start)
        end = start + length
        return ts[start:end:sampling_rate, :], target[end + target_time - 1]

    def example_set(ts, target, num, length, target_time):
        input_length = math.ceil(length / sampling_rate)
        inputs = np.zeros((num, input_length, ts.shape[1]))
        targets = np.zeros(num)
        for i in range(num):
            inp, tar = random_input(ts, length, target_time, target)
            inputs[i] = inp
            targets[i] = tar
        return inputs, targets

    train_inputs, train_targets = example_set(train_data, train_temperature, num_samples, seq_length, target_time)
    val_inputs, val_targets = example_set(val_data, val_temperature, num_samples, seq_length, target_time)
    test_inputs, test_targets = example_set(test_data, test_temperature, num_samples, seq_length, target_time)

    train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_inputs, dtype=torch.float32), torch.tensor(val_targets, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32))

    return train_dataset, val_dataset, test_dataset

# 通用训练函数
def train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, learning_rate, batch_size, model_name,logger,image_dir):
    print(f"Training {model_name} with {num_epochs} epochs, learning rate {learning_rate},batch_size {batch_size}...")
    logger.info(f"Training {model_name} with {num_epochs} epochs, learning rate {learning_rate}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
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

        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    model.load_state_dict(best_weights)

    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    logger.info(f'{model_name} Test Loss: {test_loss:.4f}')
    print(f'{model_name} Test Loss: {test_loss:.4f}')

    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.axhline(test_loss, color='red', linestyle='--', label=f'Test Loss: {test_loss:.4f}')
    plt.legend()
    plt.title(f'{model_name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(image_dir, f'{model_name}_loss_curve.png'))
    plt.show()

# 原始模型函数
def run_original_model(num_epochs, learning_rate, batch_size,logger,image_dir):
    train_dataset, val_dataset, test_dataset = load_data(logger)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = RNNModel(input_dim=14, hidden_dim=50, output_dim=1, num_layers=2)
    train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, learning_rate,batch_size, "Original_RNN",logger,image_dir)

# 加深模型函数
def run_deepened_model(num_epochs, learning_rate, batch_size,logger,image_dir):
    train_dataset, val_dataset, test_dataset = load_data(logger)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = RNNModel(input_dim=14, hidden_dim=50, output_dim=1, num_layers=15)  # 增加层数
    train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, learning_rate,batch_size, "Deepened_RNN",logger,image_dir)

# 加宽模型函数
def run_widened_model(num_epochs, learning_rate, batch_size,logger,image_dir):
    train_dataset, val_dataset, test_dataset = load_data(logger)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = RNNModel(input_dim=14, hidden_dim=100, output_dim=1, num_layers=2)  # 增加隐藏单元
    train_and_evaluate(model, train_loader, val_loader, test_loader, num_epochs, learning_rate, batch_size, "Widened_RNN",logger,image_dir)

if __name__ == "__main__":
    os.chdir('/home/gdut_students/lwb/RNN_network')
    # 设置全局日志目录
    log_dir = './logs'
    image_dir = './images_compare'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'compare.log'), mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    print("不同epoch对比实验")
    logger.info("不同epoch对比实验")
    run_original_model(num_epochs = 20, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_epoch')
    run_original_model(num_epochs = 100, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_epoch')
    run_original_model(num_epochs = 300, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_epoch')
    print("不同学习率对比实验")
    logger.info("不同学习率对比实验")
    run_original_model(num_epochs = 100, learning_rate = 0.1, batch_size = 64,logger=logger,image_dir = './images_lr')
    run_original_model(num_epochs = 100, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_lr')
    run_original_model(num_epochs = 100, learning_rate = 0.000001, batch_size = 64,logger=logger,image_dir = './images_lr')
    print("不同深度对比实验")
    logger.info("不同深度对比实验")
    run_original_model(num_epochs = 100, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_deep')
    run_deepened_model(num_epochs = 100, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_deep')
    print("不同宽度对比实验")
    logger.info("不同宽度对比实验")
    run_original_model(num_epochs = 100, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_wide')
    run_widened_model(num_epochs = 100, learning_rate = 0.001, batch_size = 64,logger=logger,image_dir = './images_wide')