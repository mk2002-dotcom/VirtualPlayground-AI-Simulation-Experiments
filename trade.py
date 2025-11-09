# trade
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


class Agent:
    def __init__(self, money):
        self.money = money
        self.saving_rate = random.uniform(0.1, 0.5)

    def trade(self, trade_amount):
        self.money += trade_amount


class World:
	def __init__(self, N_agent, init_money):
		self.agents = [Agent(init_money) for _ in range(N_agent)]

	def step(self):
		# ランダムにペアを選んでトレード
		for i in range(len(self.agents)):	
			for j in range(i + 1, len(self.agents)):
				a = self.agents[i]
				b = self.agents[j]
    
				pay = random.uniform(0, limit * (a.money / init_money))
				if random.random() < 0.5:
					giver, taker = a, b
				else:
					giver, taker = b, a
     
				if (1 - giver.saving_rate) * giver.money >= pay:
					giver.trade(-pay)
					taker.trade(pay)


# Parameters
N_agent = 30
init_money = 1000
limit = 10
N_trade = 10  # number of trades per frame

# Set up
world = World(N_agent, init_money)
fig, ax = plt.subplots(figsize=(6, 6))
x = np.arange(N_agent)
y = [a.money for a in world.agents]
bars = ax.bar(x, y, color="skyblue")
ax.set_xlim(-1, N_agent)
ax.set_ylim(0, 10000)
ax.set_xlabel("Agent Number")
ax.set_ylabel("Wealth")
ax.set_title("Wealth Distribution")

def update(frame):
    for _ in range(N_trade):
        world.step()

    new_y = [a.money for a in world.agents]

    # update bars
    for bar, height in zip(bars, new_y):
        bar.set_height(height)

    return bars


ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=True)
plt.show()