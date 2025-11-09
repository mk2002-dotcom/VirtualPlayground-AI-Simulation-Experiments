import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Linear Regression
x = torch.linspace(0, 10, 100).unsqueeze(1)
y = 3 * x + 2 +  torch.randn_like(x) * 2.0

# learning
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for i in range(1000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Epoch {i}: Loss={loss.item():.4f}")
        
# result        
W, b = model.weight.item(), model.bias.item()
print(f"W={W:.4f}, b={b:.4f}")

# plot
plt.scatter(x, y, label="Noisy Data")
plt.plot(x, model(x).detach(), color='red', label="Fitted Line")
plt.legend()
plt.show()