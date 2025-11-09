# Smoke
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parameters
N = 96             
dt = 0.08          
visc = 1e-7      
diff = 5e-5        
buoyancy_beta = 0.1  
buoyancy_alpha = 0.001 
vort_eps = 0.4         # vorticity confinement
source_density = 120.0
source_temp = 6.0

# initialize
u = np.zeros((N, N))     
v = np.zeros((N, N))     
dens = np.zeros((N, N)) 
temp = np.zeros((N, N))
p = np.zeros((N, N))

# grid coordinates (i = x index, j = y index)
j_idx, i_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')  # note: (y,x) indexing

# -------------------------
# Utility
# -------------------------
def set_bnd_scalar(x):
    # 固定（反射的）境界を簡易実装
    x[0, :] = x[1, :]
    x[-1, :] = x[-2, :]
    x[:, 0] = x[:, 1]
    x[:, -1] = x[:, -2]

def laplacian(field):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
      + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
      - 4.0 * field
    )

def diffuse(field, field0, diff_coef, iter=20):
    a = dt * diff_coef * (N * N)
    f = field
    for _ in range(iter):
        f[1:-1,1:-1] = (field0[1:-1,1:-1] + a * (
            f[1:-1,2:] + f[1:-1,:-2] + f[2:,1:-1] + f[:-2,1:-1]
        )) / (1 + 4*a)
        set_bnd_scalar(f)
    return f

# -------------------------
# Bilinear
# -------------------------
def bilinear_sample(field, xp, yp):
    xp_clipped = np.clip(xp, 0.0, N - 1.000001)
    yp_clipped = np.clip(yp, 0.0, N - 1.000001)
    i0 = np.floor(xp_clipped).astype(int)
    j0 = np.floor(yp_clipped).astype(int)
    i1 = np.clip(i0 + 1, 0, N - 1)
    j1 = np.clip(j0 + 1, 0, N - 1)
    sx = xp_clipped - i0
    sy = yp_clipped - j0

    # sample: field[j,i]
    f00 = field[j0, i0]
    f10 = field[j0, i1]
    f01 = field[j1, i0]
    f11 = field[j1, i1]
    return (1-sx)*(1-sy)*f00 + sx*(1-sy)*f10 + (1-sx)*sy*f01 + sx*sy*f11

# -------------------------
# RK4
# -------------------------
def advect_RK4(field, u_field, v_field):
    xp = i_idx.astype(float)   # shape (N,N), x coordinate (col)
    yp = j_idx.astype(float)   # y coordinate (row)

    def sample_uv(xp_s, yp_s):
        # note: bilinear_sample expects field[y,x]
        u_i = bilinear_sample(u_field, xp_s, yp_s)
        v_i = bilinear_sample(v_field, xp_s, yp_s)
        return u_i, v_i

    # k1 at (xp, yp)
    k1u, k1v = sample_uv(xp, yp)
    # k2 at (xp - 0.5*dt*k1)
    x2 = xp - 0.5 * dt * k1u * N
    y2 = yp - 0.5 * dt * k1v * N
    k2u, k2v = sample_uv(x2, y2)
    # k3
    x3 = xp - 0.5 * dt * k2u * N
    y3 = yp - 0.5 * dt * k2v * N
    k3u, k3v = sample_uv(x3, y3)
    # k4
    x4 = xp - dt * k3u * N
    y4 = yp - dt * k3v * N
    k4u, k4v = sample_uv(x4, y4)

    x_src = xp - (dt * N / 6.0) * (k1u + 2*k2u + 2*k3u + k4u)
    y_src = yp - (dt * N / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)

    field_new = bilinear_sample(field, x_src, y_src)
    set_bnd_scalar(field_new)
    return field_new

# -------------------------
# Project
# -------------------------
def project(u_field, v_field, iter=30):
    div = np.zeros_like(u_field)
    p_local = np.zeros_like(u_field)
    # divergence (centered)
    div[1:-1,1:-1] = -0.5 * ( (u_field[1:-1,2:] - u_field[1:-1,:-2]) +
                               (v_field[2:,1:-1] - v_field[:-2,1:-1]) )
    set_bnd_scalar(div)
    p_local[:] = 0.0
    for _ in range(iter):
        p_local[1:-1,1:-1] = (div[1:-1,1:-1] + p_local[1:-1,2:] + p_local[1:-1,:-2]
                              + p_local[2:,1:-1] + p_local[:-2,1:-1]) / 4.0
        set_bnd_scalar(p_local)
    # grad p
    u_field[1:-1,1:-1] -= 0.5 * (p_local[1:-1,2:] - p_local[1:-1,:-2])
    v_field[1:-1,1:-1] -= 0.5 * (p_local[2:,1:-1] - p_local[:-2,1:-1])
    set_bnd_scalar(u_field); set_bnd_scalar(v_field)

# -------------------------
# vorticity confinement
# -------------------------
def vorticity_confinement(u_field, v_field, eps=vort_eps):
    # ω = dv/dx - du/dy
    wx = (np.roll(v_field, -1, axis=1) - np.roll(v_field, 1, axis=1)) * 0.5
    wy = (np.roll(u_field, -1, axis=0) - np.roll(u_field, 1, axis=0)) * 0.5
    w = wx - wy  # scalar vorticity
    # |∇ω|
    mag_x = (np.roll(np.abs(w), -1, axis=1) - np.roll(np.abs(w), 1, axis=1)) * 0.5
    mag_y = (np.roll(np.abs(w), -1, axis=0) - np.roll(np.abs(w), 1, axis=0)) * 0.5
    mag = np.sqrt(mag_x**2 + mag_y**2) + 1e-8
    Nx = mag_x / mag
    Ny = mag_y / mag
    # force = eps * (N x ω) where N = ∇|ω|/|∇|ω||
    u_field += eps * Ny * w
    v_field -= eps * Nx * w
    set_bnd_scalar(u_field); set_bnd_scalar(v_field)

# -------------------------
# one step
# -------------------------
def step():
    global u, v, dens, temp
    # 1)
    sx = N // 2
    sy = 4
    dens[sy:sy+4, sx-2:sx+2] += source_density * dt * 0.03
    temp[sy:sy+4, sx-2:sx+2] += source_temp * dt * 0.6

    # 2)
    v += dt * (buoyancy_beta * (temp - 0.0) - buoyancy_alpha * dens)

    # 3)
    vorticity_confinement(u, v, eps=vort_eps)

    # 4) Diffusion
    u = diffuse(u, u.copy(), visc)
    v = diffuse(v, v.copy(), visc)
    dens = diffuse(dens, dens.copy(), diff)
    temp = diffuse(temp, temp.copy(), diff)

    # 5)
    project(u, v, iter=30)

    # 6)
    u = advect_RK4(u, u, v)
    v = advect_RK4(v, u, v)
    dens = advect_RK4(dens, u, v)
    temp = advect_RK4(temp, u, v)

    # 7)
    project(u, v, iter=20)

    # 8) clamp / remove NaN
    for f in (u, v, dens, temp):
        f[np.isnan(f)] = 0.0
        np.clip(f, -1e3, 1e3, out=f)

# Animation
fig, ax = plt.subplots(figsize=(6,6))
img = ax.imshow(dens, cmap="gray", origin="lower", vmin=0, vmax=5)
ax.set_title("Thermal Smoke (RK4 advection)")
ax.axis("off")

def update(frame):
    for _ in range(3):
        step()
    display = np.sqrt(np.clip(dens, 0, None))
    img.set_data(display)
    return [img]

ani = FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)
plt.show()
