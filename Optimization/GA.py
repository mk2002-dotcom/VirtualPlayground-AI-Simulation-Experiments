# Traveling salesman problem: Genetic Algorithm
# better than nearest neighbor and 2-opt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

# ===============================
# 1. Data
# ===============================
np.random.seed()
num_cities = 15
cities = np.random.rand(num_cities, 2)

def total_distance(order):
    dist = 0
    for i in range(len(order) - 1):
        dist += np.linalg.norm(cities[order[i]] - cities[order[i+1]])
    return dist

# ===============================
# 2. GA parameters
# ===============================
POP_SIZE = 80
ELITE_SIZE = 10
MUTATION_RATE = 0.2
GENERATIONS = 100

# ===============================
# 3. Initialize
# ===============================
def create_route():
    route = list(range(num_cities))
    random.shuffle(route)
    route.append(route[0]) 
    return route

def initial_population():
    return [create_route() for _ in range(POP_SIZE)]

# ===============================
# 4. Routes
# ===============================
def rank_routes(pop):
    distances = [total_distance(r) for r in pop]
    ranked = sorted(zip(distances, pop), key=lambda x: x[0])
    return ranked

# ===============================
# 5. Selection
# ===============================
def selection(ranked):
    selected = [route for _, route in ranked[:ELITE_SIZE]]
    while len(selected) < POP_SIZE:
        i, j = random.sample(range(POP_SIZE), 2)
        winner = ranked[i][1] if ranked[i][0] < ranked[j][0] else ranked[j][1]
        selected.append(winner)
    return selected

# ===============================
# 6. Crossover
# ===============================
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(1, num_cities - 1), 2))
    child = [None] * num_cities
    child[start:end] = parent1[start:end]

    fill_positions = [i for i in range(num_cities) if child[i] is None]
    fill_values = [c for c in parent2 if c not in child]

    for i, val in zip(fill_positions, fill_values):
        child[i] = val

    child.append(child[0])
    return child

# ===============================
# 7. Mutation
# ===============================
def mutate(route):
    for i in range(1, num_cities - 1):
        if random.random() < MUTATION_RATE:
            j = random.randint(1, num_cities - 2)
            route[i], route[j] = route[j], route[i]
    return route

# ===============================
# 8. Update
# ===============================
def next_generation(current_gen):
    ranked = rank_routes(current_gen)
    selected = selection(ranked)
    children = []

    for i in range(0, POP_SIZE - ELITE_SIZE):
        parent1 = selected[i]
        parent2 = selected[-i - 1]
        child = crossover(parent1, parent2)
        children.append(mutate(child))

    new_gen = selected[:ELITE_SIZE] + children
    return new_gen

population = initial_population()
history = []
for gen in range(GENERATIONS):
    population = next_generation(population)
    best_route = rank_routes(population)[0][1]
    history.append(best_route)

# ===============================
# 10. Animation
# ===============================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title("Genetic Algorithm - TSP Evolution", fontsize=14)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.scatter(cities[:,0], cities[:,1], color='red', s=50)

(line,) = ax.plot([], [], 'b-', lw=2)
text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

def update(frame):
    path = history[frame]
    line.set_data(cities[path, 0], cities[path, 1])
    dist = total_distance(path)
    text.set_text(f"Generation: {frame+1}/{GENERATIONS}\nDistance: {dist:.3f}")
    return line, text

ani = FuncAnimation(fig, update, frames=len(history), interval=200, blit=True, repeat=False)
plt.show()
