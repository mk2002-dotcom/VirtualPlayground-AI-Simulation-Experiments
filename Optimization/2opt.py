# Traveling salesman problem: 2-opt
# better than nearest neighbor
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
def total_distance(order):
    dist = 0
    for i in range(len(order) - 1):
        dist += np.linalg.norm(cities[order[i]] - cities[order[i+1]])
    return dist

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
            key=lambda x: np.linalg.norm(cities[last] - cities[x])
        )
        visited.append(next_city)
    visited.append(visited[0])
    return visited

order = nearest_neighbor(cities)

# ===============================
# 4. 2-opt steps
# ===============================
def two_opt_steps(order):
    n = len(order)
    best = order.copy()
    improved = True
    steps = []

    while improved:
        improved = False
        best_dist = total_distance(best)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                new_order = best[:i] + best[i:j][::-1] + best[j:]
                new_dist = total_distance(new_order)
                if new_dist < best_dist:
                    best = new_order
                    best_dist = new_dist
                    steps.append(best.copy())
                    improved = True
                    break
            if improved:
                break
    return steps

steps = two_opt_steps(order)
if len(steps) == 0:
    steps = [order]

all_frames = []
for step in steps:
    for i in range(1, len(step)):
        all_frames.append(step[:i+1])
        
# ===============================
# 5. Animation
# ===============================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("2-opt", fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.scatter(cities[:,0], cities[:,1], color='red', s=50, label='Cities')

for i, (x, y) in enumerate(cities):
    ax.text(x + 0.01, y + 0.01, str(i), fontsize=9)

(line,) = ax.plot([], [], 'b-', lw=2)
text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def update(frame):
    path = all_frames[frame]
    line.set_data(cities[path, 0], cities[path, 1])
    dist = total_distance(path)
    text.set_text(f"Step {frame+1}/{len(all_frames)}\nDistance: {dist:.3f}")
    return line, text

ani = FuncAnimation(
    fig, update, frames=len(all_frames), interval=300, blit=True, repeat=False
)

ax.legend()
plt.show()

