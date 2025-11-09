# Duffing oscillator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Duffing振動子のパラメータ
# -----------------------
delta = 0.2       # 減衰
alpha = -1.0      # 線形バネ
beta = 2.0        # 非線形バネ
gamma = 0.3       # 外力振幅
omega = 1.2       # 外力周波数

dt = 0.1
steps = 4000

# -----------------------
# 初期条件
# -----------------------
x = 0.5
v = 0.0
x_list = []

# -----------------------
# 時間発展（オイラー法）
# -----------------------
for i in range(steps):
    t = i * dt
    a = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)
    v += a*dt
    x += v*dt
    x_list.append(x)

t_list = np.arange(0, steps*dt, dt)

# -----------------------
# アニメーション設定
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout()

# --- 左：振動子アニメーション ---
ax1.set_xlim(-2, 2)
ax1.set_ylim(-0.5, 0.5)
ax1.set_aspect('equal')
ax1.axis('off')

(line,) = ax1.plot([], [], lw=2, color='gray')
(ball,) = ax1.plot([], [], 'o', markersize=20, color='red')

# --- 右：x(t) グラフ ---
ax2.set_xlim(0, t_list[-1])
ax2.set_ylim(min(x_list) - 0.5, max(x_list) + 0.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("x(t)")
ax2.set_title("Duffing Oscillator: Position over Time")
(time_line,) = ax2.plot([], [], color='blue', lw=1.5)

def update(frame):
    pos = x_list[frame]
    ball.set_data([pos], [0])
    # 時間グラフ更新
    time_line.set_data(t_list[:frame], x_list[:frame])
    return line, ball, time_line

ani = FuncAnimation(fig, update, frames=len(x_list), interval=20, blit=True)
plt.show()
