import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# 設定隨機種子以確保結果可複現
np.random.seed(42)
torch.manual_seed(42)

# 1. 模擬冰水機數據生成
n_samples = 10000
rf_power = np.random.choice([0, 500, 1200, 2000], size=n_samples)
laser_power = np.random.choice([0, 300, 800], size=n_samples)
heat_load = 0.8 * rf_power + 0.9 * laser_power + np.random.normal(0, 50, n_samples)
ambient_temp = 23.0 + 2.0 * np.sin(np.linspace(0, 2 * np.pi, n_samples))

water_temp = np.zeros(n_samples)
water_temp[0] = 20.0
for t in range(1, n_samples):
    cooling = 0.85 * heat_load[t-1] + 150.0 * (water_temp[t-1] - 20.0)
    water_temp[t] = water_temp[t-1] + ((heat_load[t-1] - cooling) / (50.0 * 4184.0))

df = pd.DataFrame({
    'rf_power': rf_power, 'laser_power': laser_power,
    'heat_load': heat_load, 'ambient_temp': ambient_temp, 'water_temp': water_temp
})

# 2. Fit Scaler (嚴謹防數據洩漏)
split_idx = int(len(df) * 0.8)
scaler = MinMaxScaler()
scaler.fit(df.iloc[:split_idx])
scaled_data = scaler.transform(df)

# 3. 定義 LSTM 神經網絡結構
class ChillerLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=32):
        super(ChillerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 4. 準備序列訓練集
X, y = [], []
for i in range(len(scaled_data) - 10):
    X.append(scaled_data[i:i+10])
    y.append(scaled_data[i+10, -1])

X_t = torch.FloatTensor(np.array(X[:1000]))
y_t = torch.FloatTensor(np.array(y[:1000]))

# 5. 進行模型訓練
model = ChillerLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    out = model(X_t).squeeze()
    loss = criterion(out, y_t)
    loss.backward()
    optimizer.step()

# 6. 儲存模型權重與 Scaler
torch.save(model.state_dict(), "chiller_model.pth")
joblib.dump(scaler, "scaler.pkl")
print("✅ PyTorch LSTM 模型與 Scaler 訓練儲存成功！")
