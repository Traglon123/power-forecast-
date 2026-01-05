import torch
import torch.nn as nn

class WavKANLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(WavKANLayer, self).__init__()
        # 设置可学习的参数：权重、尺度(scale)和平移(translation)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features
    
    def dog_wavelet(self, x):
        # 定义差分高斯小波变换
        return -x * torch.exp(-0.5 * x**2)
    
    def forward(self, x):
        # 核心逻辑：对输入进行平移和缩放，再经过小波函数激活
        # x 形状: (batch, in_features)
        x_reshaped = x.unsqueeze(1).expand(-1, self.out_features, -1)#unsqueeze(1)扩展维度，expand(-1, self.weight.size(0), -1)不分配内存的扩展维度
        x_scaled = (x_reshaped - self.translation) / self.scale
        phi = self.dog_wavelet(x_scaled)

        # 将结果加权求和
        return torch.sum(self.weight * phi, dim=-1)
    
class WavKANBiGRU(nn.Module):
    def __init__(self, num_features, window_size, hidden_dim=64, forecast_steps=1):
        super(WavKANBiGRU, self).__init__()
        
        # --- 支路 1: Wav-KAN 分支 ---
        # 论文逻辑：将整个窗口的数据展平，交给 Wav-KAN 提取非线性特征
        kan_in_features = num_features * window_size
        self.kan_branch = nn.Sequential(
            nn.Flatten(),
            WavKANLayer(kan_in_features, 128),  # 提取 128 维非线性特征
            nn.ReLU()
        )

        # --- 支路 2: Bi-GRU 分支 ---
        # 论文逻辑：利用双向 GRU 提取时间序列的周期性规律
        self.gru_branch = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True  # 开启双向
        )

        # --- 融合层 (Feature Fusion) ---
        # 拼接两边的特征：KAN 的 128 维 + Bi-GRU 的 128 维 (64*2)
        combined_dim = 128 + (hidden_dim * 2)

        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_steps)  # 最终输出预测值
        )

    def forward(self, x):
        # x 形状: (batch, window_size, num_features)
        
        # 1. 运行 KAN 支路
        out_kan = self.kan_branch(x)
        
        # 2. 运行 Bi-GRU 支路
        out_gru, _ = self.gru_branch(x)
        out_gru = out_gru[:, -1, :]  # 取最后一个时间步的特征
        
        # 3. 拼接特征 (Concatenate)
        # 对应图中的 Fusion 步骤
        combined = torch.cat([out_kan, out_gru], dim=1)
        
        # 4. 输出最终结果
        return self.fusion_layer(combined)
    
