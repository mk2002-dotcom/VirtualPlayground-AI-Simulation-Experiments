# Maze (A* pathfinding)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import heapq
import random

# ==== Maze ====
def generate_nonconnected_maze(rows, cols, wall_prob=0.3):
    maze = np.zeros((rows, cols), dtype=int)
    maze[0, :] = maze[-1, :] = maze[:, 0] = maze[:, -1] = 1
    for r in range(1, rows-1):
        for c in range(1, cols-1):
            if random.random() < wall_prob:
                maze[r, c] = 1
    return maze

# ==== A* pathfinding ====
def astar(maze, start, goal):
    def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    rows, cols = maze.shape
    came_from = {}
    g_score = { (r,c): float('inf') for r in range(rows) for c in range(cols) }
    g_score[start] = 0
    f_score = { (r,c): float('inf') for r in range(rows) for c in range(cols) }
    f_score[start] = heuristic(start, goal)
    open_set = [(f_score[start], start)]
    visited_order = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        visited_order.append(current)
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = current[0]+dr, current[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr,nc] == 0:
                tentative_g = g_score[current] + 1
                neighbor = (nr, nc)
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    path = []
    cur = goal
    if cur not in came_from:
        return visited_order, []
    while cur in came_from:
        path.append(cur)
        cur = came_from[cur]
    path.append(start)
    path.reverse()
    return visited_order, path

# ==== Solvable maze ====
def generate_solvable_maze(rows, cols, wall_prob=0.3, max_attempts=100):
    for _ in range(max_attempts):
        maze = generate_nonconnected_maze(rows, cols, wall_prob)
        start, goal = (1, 1), (rows-2, cols-2)
        visited, path = astar(maze, start, goal)
        if path:
            return maze, visited, path
    raise RuntimeError("No solvable maze")

# ==== Main ====
maze, visited_order, path = generate_solvable_maze(25, 25, wall_prob=0.35)
start, goal = (1, 1), (23, 23)

# ==== Animation ====
frames = []
base = np.zeros_like(maze, dtype=float)
base[maze == 1] = 0.4

for i, (r, c) in enumerate(visited_order):
    frame = base.copy()
    for vr, vc in visited_order[:i]:
        frame[vr, vc] = 0.7
    for pr, pc in path[:max(1, int(len(path) * (i / len(visited_order))))]:
        frame[pr, pc] = 1.0
    frame[start] = 0.2
    frame[goal] = 1.0
    frames.append(frame)

fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(frames[0], cmap='viridis')
ax.set_title("A* Escaping Non-Connected Maze (Guaranteed Reachable)")

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=60, repeat=False)
plt.show()
