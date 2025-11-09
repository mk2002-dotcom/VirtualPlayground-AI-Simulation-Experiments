# Lorenz attractor
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D projection registration
from matplotlib.animation import FuncAnimation


# ---------- Lorenz parameters ----------
sigma = 10.0
rho = 30.0
beta = 2.0
def lorenz(t, state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# ---------- initial ----------
y0 = [0.0, 1.0, 0.0] 
t0, t1 = 0.0, 50.0
steps = 5000
t_eval = np.linspace(t0, t1, steps)

# ---------- integral ----------
sol = solve_ivp(lorenz, (t0, t1), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

# ---------- 3D plot ----------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=0.6)
ax.set_title("Lorenz attractor")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.tight_layout()
plt.show()