# Sun, Earth, Moon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from matplotlib.animation import PillowWriter

# ===== parameters =====
G = 1.0
M_sun = 2000.0
M_earth = 80.0
M_moon = 0.01

# initial position
r_sun = np.array([0.0, 0.0])
r_earth = np.array([10.0, 0.0])
r_moon = r_earth + np.array([1.0, 0.0])

# initial velocity
v_earth = np.array([0.0, np.sqrt(G * M_sun / np.linalg.norm(r_earth))])
v_moon_rel = np.array([0.0, 0.5 * np.sqrt(G * M_earth / np.linalg.norm(r_moon - r_earth))])
v_moon = v_earth + v_moon_rel
v_sun = np.array([0.0, 0.0])

# intial state vector
y0 = np.concatenate([r_sun, r_earth, r_moon, v_sun, v_earth, v_moon])

# ===== equation of motion =====
def deriv(t, y):
    r_sun, r_earth, r_moon = y[0:2], y[2:4], y[4:6]
    v_sun, v_earth, v_moon = y[6:8], y[8:10], y[10:12]

    def acc(r1, r2, m2):
        diff = r2 - r1
        dist = np.linalg.norm(diff)
        return G * m2 * diff / dist**3

    a_sun = acc(r_sun, r_earth, M_earth) + acc(r_sun, r_moon, M_moon)
    a_earth = acc(r_earth, r_sun, M_sun) + acc(r_earth, r_moon, M_moon)
    a_moon = acc(r_moon, r_sun, M_sun) + acc(r_moon, r_earth, M_earth)

    dydt = np.concatenate([v_sun, v_earth, v_moon, a_sun, a_earth, a_moon])
    return dydt

# ===== solve =====
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)
sol = solve_ivp(deriv, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)

positions_sun = sol.y[0:2].T
positions_earth = sol.y[2:4].T
positions_moon = sol.y[4:6].T

# ===== animation =====
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-15, 15)
ax.set_ylim(-15, 15)

sun_dot, = ax.plot([], [], 'ro', ms=10)
earth_dot, = ax.plot([], [], 'bo', ms=6)
moon_dot, = ax.plot([], [], 'yo', ms=3)

earth_trace, = ax.plot([], [], 'b--', lw=0.5, alpha=0.6)
moon_trace, = ax.plot([], [], 'y--', lw=0.5, alpha=0.6)

def update(frame):
    sun_dot.set_data([positions_sun[frame, 0]], [positions_sun[frame, 1]])
    earth_dot.set_data([positions_earth[frame, 0]], [positions_earth[frame, 1]])
    moon_dot.set_data([positions_moon[frame, 0]], [positions_moon[frame, 1]])

    earth_trace.set_data(positions_earth[:frame, 0], positions_earth[:frame, 1])
    moon_trace.set_data(positions_moon[:frame, 0], positions_moon[:frame, 1])

    return sun_dot, earth_dot, moon_dot, earth_trace, moon_trace

ani = FuncAnimation(fig, update, frames=len(t_eval), interval=20, blit=True)
#plt.show()
ani.save("sun_earth_moon.gif", writer=PillowWriter(fps=30))
print("✅ GIF saved as sun_earth_moon.gif")