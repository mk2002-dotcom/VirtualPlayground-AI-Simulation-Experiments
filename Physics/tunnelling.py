# Tunnelling effect (Quantum)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import splu

# -----------------------
# Parameters
# -----------------------
hbar = 1.0
m = 1.0
Nx = 500          
dx = 0.1
dt = 0.01
Nt = 1000

x = np.linspace(0, Nx*dx, Nx)

# -----------------------
# Potential Wall
# -----------------------
V0 = 50.0         
a = 30     
V = np.zeros(Nx)
center = Nx // 2
V[center - a//2 : center + a//2] = V0

# -----------------------
# Initialize
# -----------------------
x0 = Nx*dx*0.1   
k0 = 10        
sigma = 5      
psi = np.exp(-(x-x0)**2/(2*sigma**2)) * np.exp(1j*k0*x)
psi /= np.sqrt(np.sum(np.abs(psi)**2))

# -----------------------
# Crank–Nicolson method
# -----------------------
alpha = 1j*hbar*dt/(2*m*dx**2)
main_diag = np.ones(Nx)*(1+2*alpha) + 1j*dt*V/2
off_diag = -alpha*np.ones(Nx-1)
A = diags([off_diag, main_diag, off_diag], [-1,0,1]).tocsc()
B = diags([-off_diag, np.ones(Nx)*(1-2*alpha) - 1j*dt*V/2, -off_diag], [-1,0,1]).tocsc()
solver = splu(A)

# -----------------------
# Animation
# -----------------------
fig, ax = plt.subplots()
line, = ax.plot(x, np.abs(psi)**2, lw=2, label='|ψ|²')
ax.plot(x, V/np.max(V)*0.5, 'r--', label='Potential Wall')
ax.set_xlim(0, Nx*dx)
ax.set_ylim(0, np.max(np.abs(psi)**2)*2.0)
ax.set_xlabel('x')
ax.set_ylabel('|ψ|²')
ax.set_title('Quantum Tunneling')
ax.legend()

def update(frame):
    global psi
    for _ in range(10):
        psi = solver.solve(B.dot(psi))
    line.set_ydata(np.abs(psi)**2)
    return line,

ani = FuncAnimation(fig, update, frames=Nt//10, interval=30, blit=True)
plt.show()

