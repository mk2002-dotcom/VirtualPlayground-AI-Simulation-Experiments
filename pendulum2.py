# double pendulum
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp


# parameters
L1 = 1.0   # length1
L2 = 1.0   # length2
m1 = 1.0   # mass1
m2 = 1.0   # mass2
g = 9.8

# initial state
theta1_0 = np.pi/2
omega1_0 = 0.0
theta2_0 = np.pi/2
omega2_0 = 0.0

def double_pendulum(t, y):
    theta1, omega1, theta2, omega2 = y

    delta = theta2 - theta1

    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
    denom2 = (L2/L1) * denom1

    domega1_dt = ( m2*L1*omega1**2*np.sin(delta)*np.cos(delta)
                   + m2*g*np.sin(theta2)*np.cos(delta)
                   + m2*L2*omega2**2*np.sin(delta)
                   - (m1+m2)*g*np.sin(theta1) ) / denom1

    domega2_dt = ( - m2*L2*omega2**2*np.sin(delta)*np.cos(delta)
                   + (m1+m2)*( g*np.sin(theta1)*np.cos(delta) - L1*omega1**2*np.sin(delta) - g*np.sin(theta2) ) ) / denom2

    return [omega1, domega1_dt, omega2, domega2_dt]

# integral
T = 20
t_eval = np.linspace(0, T, 1000)
sol = solve_ivp(double_pendulum, [0, T], [theta1_0, omega1_0, theta2_0, omega2_0], t_eval=t_eval)

theta1 = sol.y[0]
theta2 = sol.y[2]

# calculate x, y
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# animation
fig, ax = plt.subplots()
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect('equal')

line, = ax.plot([], [], 'o-', lw=2)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
    return line,

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=20)
plt.show()