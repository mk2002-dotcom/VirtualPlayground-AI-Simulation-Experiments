# tunnelling effect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import splu

# -----------------------
# パラメータ設定
# -----------------------
hbar = 1.0
m = 1.0
Nx = 500          # 空間の分割数
dx = 0.1
dt = 0.01
Nt = 1000

# 空間
x = np.linspace(0, Nx*dx, Nx)

# -----------------------
# ポテンシャル（中央に障壁）
# -----------------------
V0 = 50.0          # 障壁の高さ
a = 30       # 障壁の幅（単位：dx）
V = np.zeros(Nx)
center = Nx // 2
V[center - a//2 : center + a//2] = V0

# -----------------------
# 初期波束（左側から右に進む）
# -----------------------
x0 = Nx*dx*0.1   # 初期位置（左寄り）
k0 = 10           # 運動量
sigma = 5        # 波束の広がり
psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
psi /= np.sqrt(np.sum(np.abs(psi)**2))  # 正規化

# -----------------------
# Crank–Nicolson法の行列構築
# -----------------------
alpha = 1j*hbar*dt/(2*m*dx**2)
main_diag = np.ones(Nx)*(1+2*alpha) + 1j*dt*V/2
off_diag = -alpha*np.ones(Nx-1)
A = diags([off_diag, main_diag, off_diag], [-1,0,1]).tocsc()
B = diags([-off_diag, np.ones(Nx)*(1-2*alpha) - 1j*dt*V/2, -off_diag], [-1,0,1]).tocsc()
solver = splu(A)

# -----------------------
# アニメーション設定
# -----------------------
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2, lw=2, label='|ψ|²')
ax.plot(x, V/np.max(V)*0.5, 'r--', label='Potential Wall')  # ポテンシャルの位置を可視化
ax.set_xlim(0, Nx*dx)
ax.set_ylim(0, np.max(np.abs(psi)**2)*2.0)
ax.set_xlabel('x')
ax.set_ylabel('|ψ|²')
ax.set_title('Quantum Tunneling')
ax.legend()

def update(frame):
    global psi
    for _ in range(10):  # フレームごとに10ステップ進める
        psi = solver.solve(B.dot(psi))
    line.set_ydata(np.abs(psi)**2)
    return line,

ani = FuncAnimation(fig, update, frames=Nt//10, interval=30, blit=True)
plt.show()

