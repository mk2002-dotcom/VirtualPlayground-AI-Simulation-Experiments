# Market
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation

# --- Parameters ---
num_agents = 50
steps = 100
influence = 0.3
news_effect = 0.05
price_sensitivity = 0.05
avg_degree = 4

BUY = 1
SELL = -1
HOLD = 0
state_colors = {BUY:'green', SELL:'red', HOLD:'gray'}

# --- Network ---
G = nx.watts_strogatz_graph(num_agents, avg_degree, 0.1)
pos = nx.spring_layout(G, seed=42)

# --- Initialize ---
agents = np.random.choice([BUY, SELL, HOLD], size=num_agents)
price = 100
price_history = [price]

fig, (ax_net, ax_price) = plt.subplots(1,2, figsize=(12,5))
nodes = nx.draw_networkx_nodes(G, pos, node_color=[state_colors[s] for s in agents], ax=ax_net)
edges = nx.draw_networkx_edges(G, pos, ax=ax_net)
ax_net.set_title("Agent States")
ax_net.axis('off')

# --- Prices ---
ax_price.set_xlim(0, steps)
ax_price.set_ylim(50, 150)
line, = ax_price.plot([], [], color='blue')
ax_price.set_title("Stock Price")
ax_price.set_xlabel("Step")
ax_price.set_ylabel("Price")

def update(step):
    global agents, price
    
    new_agents = agents.copy()
    
    for i in range(num_agents):
        neighbors = list(G.neighbors(i))
        if neighbors:
            avg_opinion = np.mean([agents[n] for n in neighbors])
            if np.random.rand() < influence:
                if avg_opinion > 0:
                    new_agents[i] = BUY
                elif avg_opinion < 0:
                    new_agents[i] = SELL
        
        if np.random.rand() < news_effect:
            new_agents[i] = np.random.choice([BUY, SELL, HOLD])
    
    agents = new_agents
    
    crowd_effect = price_sensitivity * np.sum(agents)
    random_jump = np.random.standard_t(df=3)
    price += crowd_effect + random_jump
    price = max(price, 1)
    price_history.append(price)

    ax_net.clear()
    nx.draw_networkx_nodes(G, pos, node_color=[state_colors[s] for s in agents], ax=ax_net)
    nx.draw_networkx_edges(G, pos, ax=ax_net)
    ax_net.set_title("Agent States")
    ax_net.axis('off')
    
    line.set_data(range(len(price_history)), price_history)
    ax_price.set_xlim(0, steps)
    ax_price.set_ylim(min(price_history)-10, max(price_history)+10)
    
    return line,

ani = FuncAnimation(fig, update, frames=steps, interval=200, blit=False)
plt.tight_layout()
plt.show()
