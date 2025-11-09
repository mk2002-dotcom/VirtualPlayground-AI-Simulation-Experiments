# plasma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# シミュレーションパラメータ
# -------------------------
N = 20                # 粒子数
dt = 0.05             # タイムステップ
steps = 500           # 総ステップ数
k = 1.0               # クーロン定数（簡易単位）
Bz = 1.0              # 外部磁場の強さ（z方向のみ）

# 粒子の状態: 位置、速度、質量、電荷
pos = np.random.rand(N, 2) * 10 - 5   # [-5,5]の範囲
vel = np.random.randn(N, 2) * 0.1
mass = np.ones(N)
charge = np.random.choice([-1, 1], N)  # ±電荷

# -------------------------
# 力計算関数
# -------------------------
def compute_forces(pos, vel):
    forces = np.zeros_like(pos)
    for i in range(N):
        for j in range(i+1, N):
            # 2粒子間のベクトル
            r = pos[j] - pos[i]
            dist = np.linalg.norm(r) + 1e-2  # 0割防止
            # Coulomb force
            f = k * charge[i] * charge[j] * r / dist**3
            forces[i] += f
            forces[j] -= f
        # 磁場によるローレンツ力（v x B）
        forces[i] += charge[i] * np.array([-vel[i,1]*Bz, vel[i,0]*Bz])
    return forces

# -------------------------
# アニメーションセットアップ
# -------------------------
fig, ax = plt.subplots()
scat = ax.scatter(pos[:,0], pos[:,1], c=charge, cmap='bwr', s=100)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')

def update(frame):
    global pos, vel
    forces = compute_forces(pos, vel)
    vel += forces / mass[:,None] * dt
    pos += vel * dt
    scat.set_offsets(pos)
    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=30)
plt.show()