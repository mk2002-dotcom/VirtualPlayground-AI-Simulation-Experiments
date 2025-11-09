# birds (boids simulation)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Parameters
N = 10  # number of boids
width, height = 10.0, 10.0  # world size
max_speed = 2.0
max_force = 0.05
perception = 1.0  # neighborhood radius

np.random.seed(1)

# Initialize positions and velocities
pos = np.random.rand(N, 2) * np.array([width, height])
vel = (np.random.rand(N, 2) - 0.5) * 2.0

def limit_vector(v, max_val):
    norm = np.linalg.norm(v)
    if norm > max_val and norm > 0:
        return v / norm * max_val
    return v

def step(pos, vel):
    new_vel = vel.copy()
    for i in range(N):
        # find neighbors
        diffs = pos - pos[i]
        dists = np.linalg.norm(diffs, axis=1)
        mask = (dists > 0) & (dists < perception)
        neighbors = np.where(mask)[0]
        steer_align = np.zeros(2)
        steer_cohesion = np.zeros(2)
        steer_separation = np.zeros(2)
        
        if neighbors.size > 0:
            # Alignment: match velocity
            avg_vel = vel[neighbors].mean(axis=0)
            steer_align = avg_vel - vel[i]
            steer_align = limit_vector(steer_align, max_force)
            
            # Cohesion: steer toward average position
            avg_pos = pos[neighbors].mean(axis=0)
            desired = avg_pos - pos[i]
            steer_cohesion = desired - vel[i]
            steer_cohesion = limit_vector(steer_cohesion, max_force)
            
            # Separation: avoid crowding
            diff_vectors = pos[i] - pos[neighbors]
            inv = 1.0 / (dists[neighbors][:, None] + 1e-6)
            repulse = (diff_vectors * inv).sum(axis=0)
            steer_separation = limit_vector(repulse, max_force * 1.5)
        
        # Weight forces
        new_vel[i] += steer_align * 1.0 + steer_cohesion * 0.6 + steer_separation * 1.8
        
        # Limit speed
        new_vel[i] = limit_vector(new_vel[i], max_speed)
        
    # Update positions
    pos[:] = pos + new_vel * 0.1  # dt factor
    # Wrap around boundaries
    pos[:, 0] = np.mod(pos[:, 0], width)
    pos[:, 1] = np.mod(pos[:, 1], height)
    return new_vel

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_xticks([]); ax.set_yticks([])
scat = ax.scatter(pos[:,0], pos[:,1], s=40)

# optional: show heading as small lines
lines = [ax.plot([], [], linewidth=1)[0] for _ in range(N)]

def init():
    scat.set_offsets(pos)
    for ln in lines:
        ln.set_data([], [])
    return [scat] + lines

frames = 250

def animate(frame):
    global vel, pos
    vel = step(pos, vel)
    scat.set_offsets(pos)
    for i, ln in enumerate(lines):
        start = pos[i]
        head = pos[i] + 0.5 * limit_vector(vel[i], 1.0)
        ln.set_data([start[0], head[0]], [start[1], head[1]])
    return [scat] + lines

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=30, blit=True)
plt.show()