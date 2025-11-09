# Ants 2.Pheromone diffusion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# --- パラメータ ---
num_ants = 100
num_steps = 500
grid_size = 30
max_speed = 0.2
pheromone_deposit = 3.0      # ← 濃く（前は1.0）
evaporation_rate = 0.01      # ← 蒸発を遅く
arrival_radius = 1.0
num_foods = 5
num_nests = 2

# --- 巣・餌 ---
nest_positions = [np.array([np.random.uniform(0, grid_size),
                            np.random.uniform(0, grid_size)]) for _ in range(num_nests)]
food_positions = [np.array([np.random.uniform(0, grid_size),
                            np.random.uniform(0, grid_size)]) for _ in range(num_foods)]

# --- フェロモンマップ ---
pheromone_map = np.zeros((grid_size, grid_size))

# --- アリ初期化 ---
ants = []
for _ in range(num_ants):
    nest_idx = np.random.randint(0, num_nests)
    direction = np.random.uniform(-1, 1, 2)
    direction /= np.linalg.norm(direction)
    ants.append({'pos': nest_positions[nest_idx].copy(),
                 'carrying': False,
                 'direction': direction,
                 'nest_idx': nest_idx})

# --- アリ移動 ---
def move_ant(ant):
    x, y = ant['pos']
    nest_pos = nest_positions[ant['nest_idx']]

    if not ant['carrying']:
        # 探索アリ：ランダム移動 + フェロモン勾配
        ant['direction'] += np.random.normal(0, 0.25, 2)
        ant['direction'] /= np.linalg.norm(ant['direction'])

        ix, iy = int(x), int(y)
        if 0 <= ix < grid_size and 0 <= iy < grid_size:
            gx = (pheromone_map[min(ix + 1, grid_size - 1), iy] - pheromone_map[max(ix - 1, 0), iy])
            gy = (pheromone_map[ix, min(iy + 1, grid_size - 1)] - pheromone_map[ix, max(iy - 1, 0)])
            grad = np.array([gx, gy])
            if np.linalg.norm(grad) > 0:
                grad /= np.linalg.norm(grad)
                ant['direction'] += 0.15 * grad
                ant['direction'] /= np.linalg.norm(ant['direction'])

        ant['pos'] += ant['direction'] * max_speed

        # 餌発見
        for food_pos in food_positions:
            if np.linalg.norm(ant['pos'] - food_pos) < arrival_radius:
                ant['carrying'] = True
                break
    else:
        # 発見アリ：巣に直進 + フェロモン設置
        direction = nest_pos - ant['pos']
        dist = np.linalg.norm(direction)
        if dist > 0:
            move_dir = direction / dist * min(max_speed, dist)
            ant['pos'] += move_dir

            ix, iy = int(ant['pos'][0]), int(ant['pos'][1])
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                pheromone_map[ix, iy] += pheromone_deposit

        if dist < arrival_radius:
            ant['carrying'] = False
            direction = np.random.uniform(-1, 1, 2)
            direction /= np.linalg.norm(direction)
            ant['direction'] = direction

# --- 更新関数 ---
def update(frame):
    global pheromone_map
    for ant in ants:
        move_ant(ant)

    # 蒸発
    pheromone_map *= (1 - evaporation_rate)

    # 拡散
    pheromone_map[:] = gaussian_filter(pheromone_map, sigma=1.2)

    # 描画更新
    pher_img.set_data(pheromone_map.T)
    ant_positions = np.array([ant['pos'] for ant in ants])
    ant_scatter.set_offsets(ant_positions)
    return [pher_img, ant_scatter]

# --- 描画設定 ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.axis('off')
ax.set_aspect('equal')

# フェロモンを濃い赤で
pher_img = ax.imshow(pheromone_map.T, origin='lower', cmap='Reds', alpha=0.85, vmin=0, vmax=10)

# アリ・巣・餌
ant_scatter = ax.scatter([], [], c='black', s=40)
nest_scatter = ax.scatter([n[0] for n in nest_positions],
                          [n[1] for n in nest_positions],
                          c='blue', s=120, label="Nest")
food_scatter = ax.scatter([f[0] for f in food_positions],
                          [f[1] for f in food_positions],
                          c='limegreen', s=120, label="Food")

ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=False)
plt.show()
