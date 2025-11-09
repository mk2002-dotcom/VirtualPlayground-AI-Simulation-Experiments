# Soliton (A kind of wave)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Parameters
L = 80
N = 256
x = np.linspace(-L/2, L/2, N)
dx = x[1]-x[0]

# Iniatialize
def soliton(x, A, x0):
    return 0.5*A**2 / (np.cosh(0.5*A*(x-x0))**2)

u0 = soliton(x, 1.2, -5) + soliton(x, 0.8, 5)

# KdV equation
def kdv_rhs(t, u):
    u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)
    u_xxx = (np.roll(u, -2) - 2*np.roll(u, -1) + 2*np.roll(u, 1) - np.roll(u, 2)) / (2*dx**3)
    return (-6*u*u_x - u_xxx)

t_span = (0, 30)
t_eval = np.linspace(0, 30, 200)
sol = solve_ivp(kdv_rhs, t_span, u0, t_eval=t_eval, method='RK45')

# Animation
fig, ax = plt.subplots(figsize=(10,4))
ax.set_ylim(-0.1, 1.2)
ax.set_facecolor('black')
ax.set_title("Soliton Collision", color='white')
ax.tick_params(colors='white')

line, = ax.plot(x, sol.y[:,0], lw=2, color='cyan')

def update(i):
    y = sol.y[:,i]
    line.set_ydata(y)
    line.set_color(plt.cm.viridis(np.max(y)))
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=30)
plt.show()
