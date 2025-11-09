# virus simulation
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Agent:
    def __init__(self, x, y, infection_time = 0, status = "healthy"):
        self.x = x
        self.y = y
        self.time = infection_time
        self.status = status

    def move(self):
        self.x += random.choice([-1, 0, 1])
        self.y += random.choice([-1, 0, 1])

class World:
    def __init__(self, size, num_h, num_i, num_r):
        self.size = size
        self.agents = []
        for _ in range(num_h):
            self.agents.append(Agent(random.randrange(size), random.randrange(size)))
        for _ in range(num_i):
            self.agents.append(Agent(random.randrange(size), random.randrange(size), status="infected"))
        for _ in range(num_r):
            self.agents.append(Agent(random.randrange(size), random.randrange(size), status="recovered"))

    def step(self):
        # people move
        for c in self.agents:
            c.move()
            if c.status == "infected":
                c.time += 1
            
        # infection
        for a in [c for c in self.agents if c.status == "healthy"]:
            for b in [c for c in self.agents if c.status == "infected"]:    
                    distance = ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5
                    if distance < infection_prob:
                        if random.random() < infection_prob:
                            a.status = "infected"
        
        # recovery
        for a in [c for c in self.agents if c.status == "infected"]:
            if a.time > recovery_time:
                a.status = "recovered"
                            

# Parameters
N_healthy = 20  # number of healthy humans
N_infected = 5  # number of infected humans
size = 20  # world size
perception = 2.0  # neighborhood radius
infection_prob = 0.9
recovery_time = 20.0

# Set up
world = World(size=size, num_h=N_healthy, num_i=N_infected, num_r=0)
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-0.5, world.size - 0.5)
ax.set_ylim(-0.5, world.size - 0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Virus simulation")

scat_healthy = ax.scatter([], [], c='green', s=40, label='Healthy')
scat_infected = ax.scatter([], [], c='red', s=40, label='Infected')
scat_recovered = ax.scatter([], [], c='blue', s=40, label='Recovered')
ax.legend(loc='upper right')

def safe_offsets(x, y):
    if len(x) == 0:
        return np.empty((0,2))
    return np.column_stack((x, y))

def init():
    scat_healthy.set_offsets(np.empty((0,2)))
    scat_infected.set_offsets(np.empty((0,2)))
    scat_recovered.set_offsets(np.empty((0,2)))
    return scat_healthy, scat_infected, scat_recovered

text_stats = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     ha="left", va="top", fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

def update(frame):
    world.step()
    xh = [c.x for c in world.agents if c.status == "healthy"]
    yh = [c.y for c in world.agents if c.status == "healthy"]
    xi = [c.x for c in world.agents if c.status == "infected"]
    yi = [c.y for c in world.agents if c.status == "infected"]
    xr = [c.x for c in world.agents if c.status == "recovered"]
    yr = [c.y for c in world.agents if c.status == "recovered"]
    
    scat_healthy.set_offsets(safe_offsets(xh, yh))
    scat_infected.set_offsets(safe_offsets(xi, yi))
    scat_recovered.set_offsets(safe_offsets(xr, yr))
    
    text_stats.set_text(f"Step {frame}\nHealthy: {len(xh)}\nInfected: {len(xi)}\nRecovered: {len(xr)}")

    return scat_healthy, scat_infected, scat_recovered, text_stats

ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=True, init_func=init)
plt.show()