# single pendulum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# parameters.theta is angle, omega is angular velocity
L = 1.0 
g = 9.8 
theta0 = np.pi * 0.9 
omega0 = 0.0     

def pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = - (g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# time
T = 10
t_eval = np.linspace(0, T, 500)

# integral
sol = solve_ivp(pendulum, [0, T], [theta0, omega0], t_eval=t_eval)
theta = sol.y[0]

x = L * np.sin(theta)
y = -L * np.cos(theta)

# animation
fig, ax = plt.subplots()
ax.set_xlim(-1.2*L, 1.2*L)
ax.set_ylim(-1.2*L, 1.2*L)
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, x[frame]], [0, y[frame]])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()