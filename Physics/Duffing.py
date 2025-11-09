# Duffing oscillator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------
# Duffing parameters
# -----------------------
delta = 0.2      
alpha = -1.0      # linear
beta = 2.0        # non linear
gamma = 0.3       
omega = 1.2       

dt = 0.1
steps = 4000

# -----------------------
# Initial state
# -----------------------
x = 0.5
v = 0.0
x_list = []

# -----------------------
# Euler method
# -----------------------
for i in range(steps):
    t = i * dt
    a = -delta*v - alpha*x - beta*x**3 + gamma*np.cos(omega*t)
    v += a*dt
    x += v*dt
    x_list.append(x)

t_list = np.arange(0, steps*dt, dt)

# -----------------------
# Animation
# -----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plt.tight_layout()

# --- Left ---
ax1.set_xlim(-2, 2)
ax1.set_ylim(-0.5, 0.5)
ax1.set_aspect('equal')
ax1.axis('off')

(line,) = ax1.plot([], [], lw=2, color='gray')
(ball,) = ax1.plot([], [], 'o', markersize=20, color='red')

# --- Right ---
ax2.set_xlim(0, t_list[-1])
ax2.set_ylim(min(x_list) - 0.5, max(x_list) + 0.5)
ax2.set_xlabel("Time")
ax2.set_ylabel("x(t)")
ax2.set_title("Duffing Oscillator: Position over Time")
(time_line,) = ax2.plot([], [], color='blue', lw=1.5)

def update(frame):
    pos = x_list[frame]
    ball.set_data([pos], [0])
    time_line.set_data(t_list[:frame], x_list[:frame])
    return line, ball, time_line

ani = FuncAnimation(fig, update, frames=len(x_list), interval=20, blit=True)
plt.show()
