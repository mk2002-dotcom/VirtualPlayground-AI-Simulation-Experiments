# Spin model (2D Ising model)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# === Pareamters ===
N = 50
J = 1.0
T_start = 4.0
T_end = 1.0
cooling_steps = 200
steps_per_temp = N * N // 2

np.random.seed(0)
spins = np.random.choice([-1, 1], size=(N, N))

# --- Energy ---
def total_energy():
    E = 0
    for i in range(N):
        for j in range(N):
            S = spins[i, j]
            nb = spins[(i+1)%N, j] + spins[i, (j+1)%N]
            E += -J * S * nb
    return E / (N*N)

# --- ΔE ---
def delta_energy(i, j):
    s = spins[i, j]
    nb = spins[(i+1)%N, j] + spins[(i-1)%N, j] + spins[i, (j+1)%N] + spins[i, (j-1)%N]
    return 2 * J * s * nb

# --- Metropolis method ---
def metropolis_step(T):
    for _ in range(steps_per_temp):
        i, j = np.random.randint(0, N, size=2)
        dE = delta_energy(i, j)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1

# === Animation ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
im = ax1.imshow(spins, cmap='coolwarm', vmin=-1, vmax=1)
ax1.axis("off")

ax2.set_xlim(T_end, T_start)
ax2.set_ylim(-2, 0)
ax2.set_xlabel("Temperature T")
ax2.set_ylabel("Energy per spin")
(line,) = ax2.plot([], [], lw=2, color="black")
ax2.grid(True)

temperatures = np.linspace(T_start, T_end, cooling_steps)
energies = []

def update(frame):
    T = temperatures[frame]
    metropolis_step(T)
    E = total_energy()
    energies.append(E)

    im.set_array(spins)
    n = min(len(energies), frame + 1)
    line.set_data(temperatures[:n], energies[:n])
    ax1.set_title(f"T = {T:.2f}, E = {E:.3f}")
    return [im, line]

ani = FuncAnimation(fig, update, frames=cooling_steps, interval=100, blit=True)
plt.tight_layout()
plt.show()


