from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import WavKANBiGRU
from data_loader import get_australia_data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def final_test_with_metrics():
    # 1. 加载最佳模型
    model.load_state_dict(torch.load("best_WavKAN_model.pth"))
    model.eval()
    
    with torch.no_grad():
        # 2. 对整个测试集进行预测
        test_inputs = X_test.to(device)
        preds_norm = model(test_inputs).cpu().numpy()
        y_norm = y_test.numpy()
        
        # 3. 【关键】逆归一化：将 0-1 回到真实的负荷数值
        # 创建一个临时矩阵来辅助 scaler 还原
        def inverse_transform(scaled_val):
            # 这里的 X.shape[2] 是特征总数
            temp = np.zeros((len(scaled_val), X.shape[2]))
            temp[:, 0] = scaled_val.flatten() # 假设负荷在第 0 列
            return scaler.inverse_transform(temp)[:, 0]

        predictions = inverse_transform(preds_norm)
        actuals = inverse_transform(y_norm)
        
        # 4. 计算指标
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        print(f"\n======== 测试集性能评价 ========")
        print(f"R² (决定系数):  {r2:.4f}")
        print(f"MSE (均方误差): {mse:.2f}")
        print(f"MAE (绝对误差): {mae:.2f}")
        print(f"MAPE (百分比误差): {mape:.2f}%")
        
        # 5. 绘图展示
        plt.figure(figsize=(15, 6))
        # 取前 150 个时间步展示更清晰
        plt.plot(actuals[:150], label='Actual (Real MW)', color='#1f77b4', linewidth=2)
        plt.plot(predictions[:150], label='Predicted (WavKAN)', color='#d62728', linestyle='--')
        
        # 将指标以文本框形式加在图上
        metric_str = f'$R^2$: {r2:.4f}\nMSE: {mse:.2f}\nMAPE: {mape:.2f}%'
        plt.text(0.02, 0.95, metric_str, transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.title('Australian Load Forecasting - Model Evaluation')
        plt.xlabel('Time Steps')
        plt.ylabel('Load (MW)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.savefig("I:/power_forecast/PyWavelets/out_pictures/test_evaluation.png")

# --- 参数设置 ---
FILE_PATH = "I:\power_forecast\PyWavelets\datasets\Electricity_load_of_Australia.csv"
TARGET_COL = "Power load" # 确保这和你 CSV 里的负荷列名一致
WINDOW_SIZE = 24
BATCH_SIZE = 64
EPOCHS = 50       # 零基础建议先跑 50 轮观察
LEARNING_RATE = 0.001

# 1. 数据加载与 8:1:1 划分
X, y, scaler = get_australia_data(FILE_PATH, WINDOW_SIZE, TARGET_COL)

num_samples = len(X)
train_end = int(num_samples * 0.8)       # 前 80%
val_end = int(num_samples * 0.9)         # 中间 10%

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# 创建三个数据加载器
train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

# 2. 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = WavKANBiGRU(num_features=X.shape[2], window_size=WINDOW_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. 训练
for epoch in range(EPOCHS):
    # --- 训练阶段 ---
    model.train()
    train_loss = 0
    for batch_idx, (X_batch, y_batch) in enumerate(train_dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # --- 验证阶段 ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            val_loss += loss.item()
    avg_train = train_loss / len(train_dataloader)
    avg_val = val_loss / len(val_dataloader)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

    # 保存验证集表现最好的模型（防止过拟合）
    best_val_loss = float('inf')
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_WavKAN_model.pth")
final_test_with_metrics()

