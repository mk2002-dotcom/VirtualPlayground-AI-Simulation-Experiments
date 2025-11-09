# Precession
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# ====== parameters ======
m = 1.0       
g = 9.81      
R = 0.1       
I1 = 0.002   
I3 = 0.001

# ====== Initial state ======
# [theta, phi, psi, theta_dot, phi_dot, psi_dot]
theta0 = 0.3
phi0 = 0.0
psi0 = 0.0
theta_dot0 = 0.0
phi_dot0 = 40.0
psi_dot0 = 50.0
y0 = [theta0, phi0, psi0, theta_dot0, phi_dot0, psi_dot0]

# ====== Equation of Motion======
def top_dynamics(t, y):
    theta, phi, psi, theta_dot, phi_dot, psi_dot = y
    
    theta_ddot = (I3 * psi_dot * phi_dot * np.sin(theta) - m*g*R*np.sin(theta)) / I1
    phi_ddot = (-I3*psi_dot*theta_dot) / (I1*np.sin(theta))
    psi_ddot = 0 
    
    return [theta_dot, phi_dot, psi_dot, theta_ddot, phi_ddot, psi_ddot]

t_span = (0, 5)
t_eval = np.linspace(*t_span, 400)
sol = solve_ivp(top_dynamics, t_span, y0, t_eval=t_eval)

# ====== Shape ======
h = R
r = 0.05
u = np.linspace(0, 2*np.pi, 40)
v = np.linspace(0, 1, 10)
x = r * np.outer(np.cos(u), v)
y = r * np.outer(np.sin(u), v)
z = h * np.outer(np.ones_like(u), v)  # The top is z=0

# ====== Rotation ======
def rotate_cone(x, y, z, theta, phi, psi):
    Rz_phi = np.array([[np.cos(phi), -np.sin(phi), 0],
                       [np.sin(phi),  np.cos(phi), 0],
                       [0, 0, 1]])
    Rx_theta = np.array([[1, 0, 0],
                         [0, np.cos(theta), -np.sin(theta)],
                         [0, np.sin(theta),  np.cos(theta)]])
    Rz_psi = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi),  np.cos(psi), 0],
                       [0, 0, 1]])
    R = Rz_phi @ Rx_theta @ Rz_psi
    xyz = np.array([x.flatten(), y.flatten(), z.flatten()])
    xyz_rot = R @ xyz
    return (xyz_rot[0].reshape(x.shape),
            xyz_rot[1].reshape(y.shape),
            xyz_rot[2].reshape(z.shape))

# ====== Animation ======
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-0.15, 0.15)
ax.set_ylim(-0.15, 0.15)
ax.set_zlim(0, 0.15)
ax.set_box_aspect([1,1,1])

def update(frame):
    ax.cla()
    theta, phi, psi = sol.y[0, frame], sol.y[1, frame], sol.y[2, frame]
    x_r, y_r, z_r = rotate_cone(x, y, z, theta, phi, psi)
    ax.plot_surface(x_r, y_r, z_r, color='royalblue', alpha=0.7)
    ax.plot([-0.1, 0.1], [0,0], [0,0], 'k--', alpha=0.3)
    ax.plot([0,0], [-0.1, 0.1], [0,0], 'k--', alpha=0.3)
    ax.set_xlim(-0.15, 0.15)
    ax.set_ylim(-0.15, 0.15)
    ax.set_zlim(0, 0.15)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Precessing Top Simulation")
    return []

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=30)
plt.show()
