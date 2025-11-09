# 2D classificaition
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


N = 200
# random data
x = 4 * (torch.rand(N, 2) - 0.5)  # shape [N,2]
r = torch.sqrt(x[:,0]**2 + x[:,1]**2)
y = (r < 1.0).float().unsqueeze(1)

model = nn.Sequential(
	nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# learning
for i in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
# prediction
with torch.no_grad():
    probs = model(x)
    y_pred_label = (probs >= 0.5).float()
    
# accuracy
correct = (y_pred_label == y).sum().item()
accuracy = correct / y.size(0)
print(f"Accuracy: {accuracy:.2f}")

# plot
xx, yy = np.meshgrid(np.linspace(-2.0, 2.0, 200),
                     np.linspace(-2.0, 2.0, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

with torch.no_grad():
    probs = model(grid)
    preds = (probs >= 0.5).numpy() 
plt.figure(figsize=(5,5))
plt.contourf(xx, yy, preds.reshape(xx.shape), alpha=0.3, cmap='bwr')
plt.scatter(x[:,0], x[:,1], c=y.squeeze(), cmap='bwr')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("2D Points Classification")
plt.show()
