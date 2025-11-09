# Energy transport (linear)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pulp

# ----------------------
# Parameters
# ----------------------
np.random.seed()

plants = ['Plant1', 'Plant2', 'Plant3']
base_P_max = {'Plant1': 100, 'Plant2':80, 'Plant3': 50}

cities = ['CityA', 'CityB', 'CityC', 'CityD']
days = 30

cost = {
    ('Plant1','CityA'):2, ('Plant1','CityB'):3, ('Plant1','CityC'):1, ('Plant1','CityD'):4,
    ('Plant2','CityA'):3, ('Plant2','CityB'):2, ('Plant2','CityC'):2, ('Plant2','CityD'):3,
    ('Plant3','CityA'):4, ('Plant3','CityB'):3, ('Plant3','CityC'):2, ('Plant3','CityD'):1
}

pos = {
    'Plant1': (0,0), 'Plant2': (0,5), 'Plant3': (0,10),
    'CityA': (10,1), 'CityB': (10,4), 'CityC': (10,7), 'CityD': (10,10)
}

# Daily change
P_max_daily = []
D_daily = []
for day in range(days):
    P_max_daily.append({p: base_P_max[p]*np.random.uniform(0.9,1.1) for p in plants})
    D_daily.append({
        'CityA': np.random.randint(50,80),
        'CityB': np.random.randint(40,70),
        'CityC': np.random.randint(50,80),
        'CityD': np.random.randint(40,70)
    })

# ----------------------
# Optimization
# ----------------------
all_flows = []
all_costs = []

for day in range(days):
    D = D_daily[day]
    P_max = P_max_daily[day]
    
    prob = pulp.LpProblem(f"Day{day}", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("trans", [(i,j) for i in plants for j in cities], lowBound=0)

    prob += pulp.lpSum([cost[i,j]*x[i,j] for i in plants for j in cities])

    for i in plants:
        prob += pulp.lpSum([x[i,j] for j in cities]) <= P_max[i]

    for j in cities:
        prob += pulp.lpSum([x[i,j] for i in plants]) >= D[j]

    prob.solve()

    flow = {(i,j): x[i,j].varValue for i in plants for j in cities}
    all_flows.append(flow)
    all_costs.append(pulp.value(prob.objective))

# ----------------------
# Animation
# ----------------------
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(-1,11)
ax.set_ylim(-1,11)
ax.set_title("Energy Transport with Load Highlight")

for n,(x_pos,y_pos) in pos.items():
    color = 'red' if n in cities else 'blue'
    ax.scatter(x_pos,y_pos,color=color,s=100)
    ax.text(x_pos+0.2,y_pos,n,fontsize=10)

lines = []
for _ in range(len(plants)*len(cities)):
    line, = ax.plot([],[],lw=2)
    lines.append(line)

text = ax.text(0.02,0.95,'', transform=ax.transAxes)

def update(day):
    flow = all_flows[day]
    idx = 0
    for i in plants:
        for j in cities:
            x0,y0 = pos[i]
            x1,y1 = pos[j]
            lines[idx].set_data([x0,x1],[y0,y1])
            
            lw = flow[(i,j)] / 10
            lines[idx].set_linewidth(max(lw,0.5))
            
            demand = D_daily[day][j]
            load_ratio = flow[(i,j)] / demand if demand>0 else 0
            if flow[(i,j)] < demand*0.9:
                color = 'red'  
            elif load_ratio > 1.2:
                color = 'orange'  
            else:
                color = 'green'  
            lines[idx].set_color(color)
            idx +=1

    demand_str = ", ".join([f"{c}:{D_daily[day][c]}" for c in cities])
    generation_str = ", ".join([f"{p}:{P_max_daily[day][p]:.1f}" for p in plants])
    text.set_text(f"Day {day+1}\nTotal cost: {all_costs[day]:.1f}\nDemand: {demand_str}\nGeneration: {generation_str}")
    return lines + [text]

ani = FuncAnimation(fig, update, frames=days, interval=600, blit=True, repeat=False)
plt.show()
