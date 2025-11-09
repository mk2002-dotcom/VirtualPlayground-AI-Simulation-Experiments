# Ants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- パラメータ ---
num_ants = 50
num_steps = 500
grid_size = 30
max_speed = 0.2
pheromone_deposit = 1.0
evaporation_rate = 0.02
arrival_radius = 0.5
num_foods = 7
num_nests = 3

# --- 巣・餌 ---
nest_positions = [np.array([np.random.uniform(0, grid_size),
                            np.random.uniform(0, grid_size)]) for _ in range(num_nests)]
food_positions = [np.array([np.random.uniform(0, grid_size),
                            np.random.uniform(0, grid_size)]) for _ in range(num_foods)]

# --- フェロモン初期化 ---
pheromone_list = []  # 各要素: [x, y, intensity]

# --- アリ初期化 ---
ants = []
for _ in range(num_ants):
    nest_idx = np.random.randint(0, num_nests)
    direction = np.random.uniform(-1,1,2)
    direction /= np.linalg.norm(direction)
    ants.append({'pos': nest_positions[nest_idx].copy(),
                 'carrying': False,
                 'direction': direction,
                 'nest_idx': nest_idx})

# --- フェロモン設置 ---
def deposit_pheromone_continuous(ant):
    pheromone_list.append([ant['pos'][0], ant['pos'][1], pheromone_deposit])

# --- アリ移動 ---
def move_ant(ant):
    x, y = ant['pos']
    nest_pos = nest_positions[ant['nest_idx']]
    
    if not ant['carrying']:
        # 探索アリ：ランダム移動 + 弱めのフェロモン勾配誘導
        ant['direction'] += np.random.normal(0,0.25,2)
        ant['direction'] /= np.linalg.norm(ant['direction'])
        for ph in pheromone_list:
            dx = ph[0] - x
            dy = ph[1] - y
            dist = np.hypot(dx, dy)
            if 0 < dist < 0.5:  # 勾配距離を狭める
                grad = np.array([dx, dy])/dist
                ant['direction'] += 0.15 * grad  # 弱め
        ant['direction'] /= np.linalg.norm(ant['direction'])
        ant['pos'] += ant['direction'] * max_speed
        
        # 餌発見
        for food_pos in food_positions:
            if np.linalg.norm(ant['pos'] - food_pos) < arrival_radius:
                ant['carrying'] = True
                break
    else:
        # 発見アリ：所属巣に直進
        direction = nest_pos - ant['pos']
        dist = np.linalg.norm(direction)
        if dist > 0:
            move_dir = direction / dist * min(max_speed, dist)
            ant['pos'] += move_dir
            deposit_pheromone_continuous(ant)
        
        # 巣到達
        if dist < arrival_radius:
            ant['carrying'] = False
            direction = np.random.uniform(-1,1,2)
            direction /= np.linalg.norm(direction)
            ant['direction'] = direction

# --- 更新関数 ---
def update(frame):
    global pheromone_list
    
    for ant in ants:
        move_ant(ant)
    
    # フェロモン蒸発
    new_pheromone = []
    for ph in pheromone_list:
        ph[2] *= (1 - evaporation_rate)
        if ph[2] > 0.001:
            new_pheromone.append(ph)
    pheromone_list[:] = new_pheromone
    
    # 描画
    if pheromone_list:
        pher_positions = np.array([[ph[0], ph[1]] for ph in pheromone_list])
        pher_alpha = np.array([ph[2] for ph in pheromone_list])
        pher_scatter.set_offsets(pher_positions)
        pher_scatter.set_alpha(pher_alpha)
    else:
        pher_scatter.set_offsets(np.zeros((0,2)))
    
    if ants:
        ant_positions = np.array([ant['pos'] for ant in ants])
    else:
        ant_positions = np.zeros((0,2))
    
    ant_scatter.set_offsets(ant_positions)
    
    return [pher_scatter, ant_scatter]

# --- アニメーション ---
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_facecolor('white')
ax.set_aspect('equal')
ax.axis('off')

pher_scatter = ax.scatter([], [], c='red', s=20, alpha=0.6)
ant_scatter = ax.scatter([], [], c='black', s=50)
nest_scatter = ax.scatter([n[0] for n in nest_positions],
                          [n[1] for n in nest_positions],
                          c='blue', s=100)
food_scatter = ax.scatter([f[0] for f in food_positions],
                          [f[1] for f in food_positions],
                          c='green', s=100)

ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=True)
plt.show()
