# boiling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# パラメータ
# ------------------------
N = 100
dt = 0.01
t_max = 30
T_init = 0.05
T_max = 0.5
heating_duration = 200
sigma = 0.05     # LJ距離スケール
epsilon = 0.01   # LJ強さ
cutoff = 0.2     # 相互作用カットオフ距離

# ------------------------
# 初期位置
# ------------------------
pos = np.zeros((N,2))
grid_size = int(np.ceil(np.sqrt(N)))
spacing = 0.1
for i in range(N):
    pos[i,0] = (i % grid_size) * spacing
    pos[i,1] = (i // grid_size) * spacing

vel = np.zeros((N,2))

# ------------------------
# 描画
# ------------------------
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-0.1, grid_size*spacing + 0.1)
ax.set_ylim(-0.1, grid_size*spacing + 0.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('black')

scat = ax.scatter([], [], s=100, color='cyan')

# ------------------------
# 力計算関数
# ------------------------
def compute_forces(pos):
    forces = np.zeros_like(pos)
    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            dist = np.linalg.norm(delta)
            if dist < cutoff and dist>0:
                F = 24*epsilon*((2*(sigma/dist)**12) - ((sigma/dist)**6)) / dist
                forces[i] += F * delta
                forces[j] -= F * delta
    return forces

# ------------------------
# 更新関数
# ------------------------
def update(frame):
    global pos, vel

    # 温度に応じた振動
    if frame < heating_duration:
        temp_factor = T_init + (T_max-T_init) * frame / heating_duration
    else:
        temp_factor = T_max

    vel += (np.random.rand(N,2)-0.5)*5*temp_factor*dt

    # 分子間力
    forces = compute_forces(pos)
    vel += forces*dt

    # 位置更新
    pos += vel*dt

    # 描画更新
    scat.set_offsets(pos)
    ax.set_title("Microscopic Bubble Formation (Free Particles, Vibration Only)", color='white')
    return scat,

ani = FuncAnimation(fig, update, frames=int(t_max/dt), interval=30, blit=False)
plt.show()
