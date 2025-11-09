# Traveling salesman problem: Nearest Neighbor
# simplest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===============================
# 1. Data
# ===============================
np.random.seed()
num_cities = 15
cities = np.random.rand(num_cities, 2)

# ===============================
# 2. Distance
# ===============================
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

# ===============================
# 3. Nearest Neighbor
# ===============================
def nearest_neighbor(cities):
    n = len(cities)
    visited = [0]
    while len(visited) < n:
        last = visited[-1]
        next_city = min(
            [i for i in range(n) if i not in visited],
            key=lambda x: distance(cities[last], cities[x])
        )
        visited.append(next_city)

    visited.append(visited[0])
    return visited

order = nearest_neighbor(cities)

# ===============================
# 4. Animation
# ===============================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("Nearest Neighbor TSP Simulation", fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.scatter(cities[:, 0], cities[:, 1], color='red', s=50, label='Cities')

for i, (x, y) in enumerate(cities):
    ax.text(x + 0.01, y + 0.01, str(i), fontsize=10)

(line,) = ax.plot([], [], 'b-', lw=2, label='Path')

def update(frame):
    # frame 0, 1, 2, ...
    path = order[:frame + 1]
    line.set_data(cities[path, 0], cities[path, 1])
    return (line,)

ani = FuncAnimation(
    fig, update, frames=len(order), interval=800, blit=True, repeat=False
)

ax.legend()
plt.show()
