import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ---- 1. データ作成 ----
x = np.linspace(-2*np.pi, 2*np.pi, 200)
y = np.sin(x) + 0.2*np.random.randn(*x.shape) 
x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# ---- 2. ネットワーク定義 ----
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)

net = Net()

# ---- 3. 損失関数と最適化 ----
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.005)

# ---- 4. 学習ループ ----
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    #if (epoch+1) % 500 == 0:
    #   print(f"Epoch [{epoch+1}/2000] Loss: {loss.item():.4f}")

# ---- 5. 予測と可視化 ----
net.eval()
with torch.no_grad():
    y_pred = net(x_train).numpy()

plt.figure()
plt.scatter(x, y, label="Noisy data", s=15)
plt.plot(x, y_pred, label="NN fit", color='red')
plt.title("Neural Network Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()