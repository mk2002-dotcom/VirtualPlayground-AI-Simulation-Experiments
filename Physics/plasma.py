# Plasma particles
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------
# parameters
# -------------------------
N = 20                
dt = 0.05            
steps = 500           
k = 1.0              
Bz = 1.0              

pos = np.random.rand(N, 2) * 10 - 5
vel = np.random.randn(N, 2) * 0.1
mass = np.ones(N)
charge = np.random.choice([-1, 1], N)

# -------------------------
# Forces
# -------------------------
def compute_forces(pos, vel):
    forces = np.zeros_like(pos)
    for i in range(N):
        for j in range(i+1, N):

            r = pos[j] - pos[i]
            dist = np.linalg.norm(r) + 1e-2
            # Coulomb force
            f = k * charge[i] * charge[j] * r / dist**3
            forces[i] += f
            forces[j] -= f
            
        # Lorentz force
        forces[i] += charge[i] * np.array([-vel[i,1]*Bz, vel[i,0]*Bz])
    return forces

# -------------------------
# Animation
# -------------------------
fig, ax = plt.subplots()
scat = ax.scatter(pos[:,0], pos[:,1], c=charge, cmap='bwr', s=100)
ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect('equal')

def update(frame):
    global pos, vel
    forces = compute_forces(pos, vel)
    vel += forces / mass[:,None] * dt
    pos += vel * dt
    scat.set_offsets(pos)
    return scat,

ani = FuncAnimation(fig, update, frames=steps, interval=30)
plt.show()