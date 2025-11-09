# Inventory management
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ----------------------
# 1. Parameters
# ----------------------
np.random.seed()

days = 30
num_products = 3

params = {
    "initial_inventory": [50, 40, 60],
    "daily_demand_mean": [5, 7, 4],
    "daily_demand_std": [2, 3, 1],
    "reorder_point": [20, 25, 30],
    "reorder_qty": [40, 50, 30],
    "max_inventory": [100, 100, 100],
    "lead_time": [2, 3, 1]
}

# ----------------------
# 2. Simulation
# ----------------------
inventory = [params["initial_inventory"].copy()]
pending_orders = [[] for _ in range(num_products)]

# Demand
demand_list = np.zeros((days, num_products), dtype=int)
for p in range(num_products):
    daily = np.random.normal(params["daily_demand_mean"][p],
                             params["daily_demand_std"][p], days)
    daily = np.clip(daily, 0, None)
    demand_list[:, p] = daily.astype(int)

for day in range(days):
    current_stock = inventory[-1].copy()

    for p in range(num_products):
        demand = demand_list[day, p]
        current_stock[p] = max(current_stock[p] - demand, 0)

    for p in range(num_products):
        for order in pending_orders[p]:
            if order[0] == day:
                current_stock[p] += order[1]
        pending_orders[p] = [o for o in pending_orders[p] if o[0] > day]

        if current_stock[p] <= params["reorder_point"][p]:
            qty = min(params["reorder_qty"][p],
                      params["max_inventory"][p] - current_stock[p])
            arrival_day = day + params["lead_time"][p]
            pending_orders[p].append((arrival_day, qty))

    inventory.append(current_stock.copy())

inventory = np.array(inventory)

# ----------------------
# 3. Animation
# ----------------------
fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(0, days)
ax.set_ylim(0, max(params["max_inventory"])+100)
ax.set_xlabel("Day")
ax.set_ylabel("Inventory Level")
ax.set_title("Multi-Product Inventory Simulation")

colors = ['b', 'g', 'orange']
lines = [ax.plot([], [], lw=2, color=c, label=f'Product {i}')[0] 
         for i, c in enumerate(colors)]
ax.legend()

text = ax.text(0.02, 0.9, '', transform=ax.transAxes)

def update(day):
    for i, line in enumerate(lines):
        line.set_data(range(day+1), inventory[:day+1, i])
    stock_str = ", ".join([f"P{i}:{inventory[day,i]}" for i in range(num_products)])
    text.set_text(f"Day {day}\nStock: {stock_str}")
    return lines + [text]

ani = FuncAnimation(fig, update, frames=days+1, interval=400, blit=True, repeat=False)
plt.show()
