# Metronome
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# -----------------------------
# Parameters
# -----------------------------
N = 5
L = 1.0
m = 0.1
M = 0.5
g = 9.8

# Initialize
theta0 = np.random.uniform(-0.1, 0.1, N)
omega0 = np.zeros(N)                       
X0 = 0.0                                 
VX0 = 0.0                                 
y0 = np.concatenate([theta0, omega0, [X0, VX0]])

# -----------------------------
# Equation of Motion
# -----------------------------
def metronome_system(t, y):
    theta = y[:N]
    omega = y[N:2*N]
    X, VX = y[-2:]
    
    cos_theta = np.cos(theta)
    
    denom = M + m * np.sum(cos_theta**2)
    X_acc = - m * np.sum(omega**2 * L * np.sin(theta) + (g / L) * np.sin(theta) * cos_theta) / denom
    
    theta_acc = - (g / L) * np.sin(theta) - X_acc * cos_theta / L
    
    return np.concatenate([omega, theta_acc, [VX, X_acc]])

# -----------------------------
# Simulation
# -----------------------------
T = 100
steps = 4000
t_eval = np.linspace(0, T, steps)

sol = solve_ivp(metronome_system, [0, T], y0, t_eval=t_eval)
theta_sol = sol.y[:N]
X_sol = sol.y[-2]       

x_met = L * np.sin(theta_sol)
y_met = -L * np.cos(theta_sol)

# -----------------------------
# Animation
# -----------------------------
fig, ax = plt.subplots()
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-1.5, 0.1)
ax.set_aspect('equal')

lines = [ax.plot([], [], 'o-', lw=2)[0] for _ in range(N)]
platform, = ax.plot([], [], 'k-', lw=5)

def init():
    for line in lines:
        line.set_data([], [])
    platform.set_data([], [])
    return lines + [platform]

def update(frame):
    X = X_sol[frame]
    platform.set_data([-1+X, 1+X], [0, 0])

    for i in range(N):
        x = np.array([X, X + x_met[i, frame]])
        y = np.array([0, y_met[i, frame]]) + i*0.05
        lines[i].set_data(x, y)
    return lines + [platform]

ani = FuncAnimation(fig, update, frames=steps,
                    init_func=init, blit=True, interval=10)
plt.show()