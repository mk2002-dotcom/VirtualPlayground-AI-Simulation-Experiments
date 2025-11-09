# boiling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------
# parameters
# ------------------------
N = 100
dt = 0.01
t_max = 30
T_init = 0.05
T_max = 0.5
heating_duration = 200
sigma = 0.05    
epsilon = 0.01   
cutoff = 0.2 

# ------------------------
# Initial state
# ------------------------
pos = np.zeros((N,2))
grid_size = int(np.ceil(np.sqrt(N)))
spacing = 0.1
for i in range(N):
    pos[i,0] = (i % grid_size) * spacing
    pos[i,1] = (i // grid_size) * spacing

vel = np.zeros((N,2))

# ------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-0.1, grid_size*spacing + 0.1)
ax.set_ylim(-0.1, grid_size*spacing + 0.1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('black')

scat = ax.scatter([], [], s=100, color='cyan')

# ------------------------
# Forces
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
# Update
# ------------------------
def update(frame):
    global pos, vel

    if frame < heating_duration:
        temp_factor = T_init + (T_max-T_init) * frame / heating_duration
    else:
        temp_factor = T_max

    vel += (np.random.rand(N,2)-0.5)*5*temp_factor*dt

    forces = compute_forces(pos)
    vel += forces*dt

    pos += vel*dt

    scat.set_offsets(pos)
    ax.set_title("Microscopic Bubble Formation (Free Particles, Vibration Only)", color='white')
    return scat,

ani = FuncAnimation(fig, update, frames=int(t_max/dt), interval=30, blit=False)
plt.show()
