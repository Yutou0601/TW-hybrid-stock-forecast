import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np

# Params
HIDDEN_DIM = 64     
FUSION_DIM = 64     

# ==========================================
# [Mamba 適配層] 
# ==========================================
HAS_MAMBA = False 
try:
    from mamba import MambaBlock, MambaConfig
    HAS_MAMBA = True
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            config = MambaConfig(d_model=d_model, n_layers=1, d_state=d_state, d_conv=d_conv, expand_factor=expand, use_cuda=False)
            self.inner = MambaBlock(config)
        def forward(self, x): return self.inner(x)
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x): raise NotImplementedError("Mamba module not loaded.")

# ==========================================
# 1. 新聞編碼器 (News Encoder: Bi-RCNN)
# 🚀 適配 FinBERT 768 維 + 雙向 LSTM
# ==========================================
class NewsRCNN(nn.Module):
    def __init__(self, input_dim=768, cnn_filters=64, kernel_size=3, lstm_hidden=64):
        super(NewsRCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 雙向結構提升特徵表達力
        self.lstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(lstm_hidden * 2)
        self.fc = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.dropout = nn.Dropout(0.3) 

    def forward(self, x):
        x = x.transpose(1, 2) 
        x = self.pool(self.relu(self.conv1d(x)))
        x = x.transpose(1, 2) 
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.dropout(self.fc(out))

# ==========================================
# 2. 股價編碼器 (LSTM / SAMBA)
# ==========================================
class StockLSTM(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, batch_first=True)
    def forward(self, x): return self.lstm(x)[0]

class AdaptiveGraphConv(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, 10))
    def forward(self, x):
        adj = F.softmax(F.relu(torch.mm(self.node_embedding, self.node_embedding.transpose(0, 1))), dim=1)
        return torch.matmul(x, adj) + x

class SAMBA_Encoder(nn.Module):
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.mamba_fwd = Mamba(d_model=hidden_dim)
        self.mamba_bwd = Mamba(d_model=hidden_dim)
        self.agc = AdaptiveGraphConv(num_nodes=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        x = self.embedding(x) 
        fwd = self.mamba_fwd(x)
        bwd = self.mamba_bwd(torch.flip(x, [1])) 
        x_time = fwd + torch.flip(bwd, [1]) 
        return self.norm(self.agc(x_time))

# ==========================================
# 3. 閘門融合層 (Gated Fusion)
# 🚀 負偏置初始化，強制啟動新聞分支
# ==========================================
class GatedFusion(nn.Module):
    def __init__(self, stock_dim, news_dim, fusion_dim=64):
        super(GatedFusion, self).__init__()
        self.project_stock = nn.Sequential(nn.Linear(stock_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        self.project_news = nn.Sequential(nn.Linear(news_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.ReLU())
        
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )
        
        # 🚀 關鍵修改：初始化負向偏置。這會讓 z 初始值偏向新聞模態 (z < 0.5)
        nn.init.constant_(self.gate_net[-2].bias, -1.0) 
        self.predictor = nn.Linear(fusion_dim, 1)

    def forward(self, stock_vec, news_vec):
        if stock_vec.dim() == 3:
            seq_len = stock_vec.size(1)
            news_vec_expanded = news_vec.unsqueeze(1).expand(-1, seq_len, -1)
            
            h_s = self.project_stock(stock_vec)
            h_n = self.project_news(news_vec_expanded)
            
            combined = torch.cat([h_s, h_n], dim=-1)
            z = self.gate_net(combined) 
            
            # 訓練時加入擾動，增加探索
            if self.training:
                noise = (torch.rand_like(z) - 0.5) * 0.02
                z = torch.clamp(z + noise, 0.0, 1.0)
            
            h_fused = z * h_s + (1 - z) * h_n
            prediction = self.predictor(h_fused[:, -1, :]) 
            return prediction, z.squeeze(-1)
        else:
            h_s = self.project_stock(stock_vec)
            h_n = self.project_news(news_vec)
            z = self.gate_net(torch.cat([h_s, h_n], dim=-1))
            return self.predictor(z * h_s + (1-z) * h_n), z

# ==========================================
# 4. 統一實驗模型 (Unified Model)
# ==========================================
class UnifiedExperimentModel(nn.Module):
    def __init__(self, num_price_features, model_type='lstm', num_news_features=768):
        super().__init__()
        if model_type == 'samba':
            self.stock_encoder = SAMBA_Encoder(num_features=num_price_features, hidden_dim=HIDDEN_DIM)
        else:
            self.stock_encoder = StockLSTM(num_features=num_price_features, hidden_dim=HIDDEN_DIM)
            
        self.news_encoder = NewsRCNN(input_dim=num_news_features, lstm_hidden=HIDDEN_DIM)
        self.fusion_layer = GatedFusion(stock_dim=HIDDEN_DIM, news_dim=HIDDEN_DIM, fusion_dim=FUSION_DIM)

    def forward(self, x_price, x_text):
        s_vec = self.stock_encoder(x_price)
        n_vec = self.news_encoder(x_text)
        return self.fusion_layer(s_vec, n_vec)

# ==========================================
# 5. 繪圖工具
# ==========================================
def plot_results(history, filename="hero_chart.png"):
    save_dir="reports/figures"
    os.makedirs(save_dir, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:orange'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test MSE', color=color)
    ax1.plot(history['test_loss'], color=color, linewidth=2, label='Test MSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Gate Value z (1=Stock, 0=News)', color=color)
    avg_gate = np.array(history['avg_gate']).flatten()
    ax2.plot(avg_gate, color=color, linewidth=2, linestyle='--', label='Avg Gate z')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1)
    
    plt.title(f'Training Progress: {filename}')
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()