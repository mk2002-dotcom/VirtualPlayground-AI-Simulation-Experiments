# interference
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# 画面サイズと波の設定
nx, ny = 200, 200  # グリッドサイズ
x = np.linspace(-5, 5, nx)
y = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x, y)

# 波源の位置
src1 = (-2.0, 0)
src2 = (2.0, 0)

# 波のパラメータ
k = 5.0       # 波数（波長の逆数に対応）
omega = 1.0   # 角速度（速さ）
t = 0.0

# 初期位相を0にしておく
r1 = np.sqrt((X - src1[0])**2 + (Y - src1[1])**2)
r2 = np.sqrt((X - src2[0])**2 + (Y - src2[1])**2)
Z = np.sin(k*r1) + np.sin(k*r2)

# 図の設定
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
img = ax.imshow(Z, extent=[-5,5,-5,5], cmap='Blues', vmin=-2, vmax=2)

# アニメーション関数
def animate(frame):
    t = frame * 0.3
    r1 = np.sqrt((X - src1[0])**2 + (Y - src1[1])**2)
    r2 = np.sqrt((X - src2[0])**2 + (Y - src2[1])**2)
    Z = np.sin(k*r1 - omega*t) + np.sin(k*r2 - omega*t)
    img.set_data(Z)
    return [img]

# アニメーション作成
anim = animation.FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.show()