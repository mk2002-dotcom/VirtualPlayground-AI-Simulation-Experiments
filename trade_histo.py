# trade (histogram)
# trade histogram
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
N_agent = 100
init_money = 1000
limit = 10
N_trade = 10
N_bins = 20  # ヒストグラムのビン数

# Set up
world = World(N_agent, init_money)
fig, ax = plt.subplots(figsize=(6,6))
hist_values, bins, patches = ax.hist([a.money for a in world.agents], bins=N_bins, color="skyblue")

def update(frame):
    for _ in range(N_trade):
        world.step()
    ax.cla()  # 前のヒストグラムを消す
    ax.set_xlim(0, init_money * 10)
    ax.set_ylim(0, N_agent)
    ax.set_title("Wealth Distribution")
    ax.set_xlabel("Wealth")
    money_list = [a.money for a in world.agents]
    ax.hist(money_list, bins=N_bins, color="skyblue")
    return []

ani = animation.FuncAnimation(fig, update, frames=100, interval=200, blit=False)
plt.show()
