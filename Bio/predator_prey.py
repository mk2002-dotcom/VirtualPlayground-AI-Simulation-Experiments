# predators and prey
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Creature:
    def __init__(self, x, y, gender, lifetime):
        self.x = x
        self.y = y
        self.gender = gender
        self.lifetime = lifetime
        self.age = 0
        self.alive = True

    def move(self, size):
        if not self.alive:
            return
        self.x = (self.x + random.choice([-1, 0, 1])) % size
        self.y = (self.y + random.choice([-1, 0, 1])) % size
        self.age += 1
        # natural death
        if self.age >= self.lifetime:
            self.alive = False

class A(Creature):
    color = "green"

class B(Creature):
    color = "red"
    def __init__(self, x, y, gender, lifetime, hunger_limit):
        super().__init__(x, y, gender, lifetime)
        self.hunger = 0
        self.hunger_limit = hunger_limit

    def eat(self):
        self.hunger = 0

    def update_hunger(self):
        self.hunger += 1
        # death from starvation
        if self.hunger >= self.hunger_limit:
            self.alive = False

class World:
    def __init__(self, size, num_A, num_B):
        self.size = size
        self.creatures = []
        for _ in range(num_A):
            self.creatures.append(A(random.randrange(size), random.randrange(size),
                                    random.choice(["M", "F"]), lifetime=30))
        for _ in range(num_B):
            self.creatures.append(B(random.randrange(size), random.randrange(size),
                                    random.choice(["M", "F"]), lifetime=35, hunger_limit=20))

    def step(self):
        # A and B move
        for c in self.creatures:
            c.move(self.size)

        # B eats A
        for a in [c for c in self.creatures if isinstance(c, A) and c.alive]:
            for b in [c for c in self.creatures if isinstance(c, B) and c.alive]:
                if a.x == b.x and a.y == b.y:
                    a.alive = False
                    b.eat()

        # a and b reproduce
        newborns = []
        for species in (A, B):
            males = [c for c in self.creatures if isinstance(c, species) and c.gender == "M" and c.alive]
            females = [c for c in self.creatures if isinstance(c, species) and c.gender == "F" and c.alive]
            for f in females:
                for m in males:
                    if f.x == m.x and f.y == m.y:
                        if species == A:
                            newborns.append(A(f.x, f.y, random.choice(["M", "F"]), lifetime=f.lifetime))
                        else:
                            newborns.append(B(f.x, f.y, random.choice(["M", "F"]), lifetime=f.lifetime, hunger_limit=5))
                        break
        self.creatures.extend(newborns)

        # hunger
        for b in [c for c in self.creatures if isinstance(c, B)]:
            b.update_hunger()

        # update
        self.creatures = [c for c in self.creatures if c.alive]

# --- animation setup ---
world = World(size=40, num_A=20, num_B=20)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-0.5, world.size - 0.5)
ax.set_ylim(-0.5, world.size - 0.5)
ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Predator-Prey")

scat_A = ax.scatter([], [], c='green', s=40, label='Prey A')
scat_B = ax.scatter([], [], c='red', s=40, label='Predator B')
ax.legend(loc='upper right')

def safe_offsets(x, y):
    if len(x) == 0:
        return np.empty((0,2))
    return np.column_stack((x, y))

def init():
    scat_A.set_offsets(np.empty((0,2)))
    scat_B.set_offsets(np.empty((0,2)))
    return scat_A, scat_B

text_stats = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                     ha="left", va="top", fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

def update(frame):
    world.step()
    xA = [c.x for c in world.creatures if isinstance(c, A)]
    yA = [c.y for c in world.creatures if isinstance(c, A)]
    xB = [c.x for c in world.creatures if isinstance(c, B)]
    yB = [c.y for c in world.creatures if isinstance(c, B)]

    scat_A.set_offsets(safe_offsets(xA, yA))
    scat_B.set_offsets(safe_offsets(xB, yB))

    text_stats.set_text(f"Step {frame}\nA: {len(xA)}\nB: {len(xB)}")

    return scat_A, scat_B, text_stats

ani = animation.FuncAnimation(fig, update, frames=300, interval=200, blit=True, init_func=init)
plt.show()
