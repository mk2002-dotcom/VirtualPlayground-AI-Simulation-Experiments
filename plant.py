# growing plant
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import numpy as np
import random

# --- L-systemルール ---
rules = {"X":[("F[+X][-X]FX",1.0)], "F":[("FF",1.0)]}

def apply_rules(axiom, rules, max_iter):
    expansions = [axiom]
    for _ in range(max_iter):
        new_str = "".join(rules.get(ch,[(ch,1.0)])[0][0] for ch in expansions[-1])
        expansions.append(new_str)
    return expansions

def render_segments(axiom, angle=25, base_length=10, shrink=0.8):
    stack = []
    x, y = 0.0, 0.0
    heading = np.pi/2
    segments = []
    widths = []
    leaves = []

    for cmd in axiom:
        depth = len(stack)
        length = base_length * (shrink**depth)
        width = 2.0*(shrink**depth)

        if cmd=="F":
            nx = x + np.cos(heading)*length
            ny = y + np.sin(heading)*length
            segments.append(((x,y),(nx,ny)))
            widths.append(width)
            # ランダムに枝上に葉を追加
            if random.random()<0.3:  # 葉が付く確率
                fx = x + random.uniform(0,1)*length
                fy = y + random.uniform(0,1)*length
                leaf_size = 5*(shrink**depth)*(0.7+0.6*random.random())
                leaves.append((fx,fy,leaf_size))
            x, y = nx, ny
        elif cmd=="+": heading -= np.radians(angle)
        elif cmd=="-": heading += np.radians(angle)
        elif cmd=="[":
            stack.append((x,y,heading))
        elif cmd=="]":
            leaves.append((x,y,5*(shrink**depth))) # 枝末端にも葉
            x, y, heading = stack.pop()

    return np.array(segments), widths, leaves

# ---------- パラメータ ----------
axiom = "X"
max_iter = 6
angle = 25
base_length = 3
shrink = 0.9

expansions = apply_rules(axiom, rules, max_iter)
segments_per_iter = []
widths_per_iter = []
leaves_per_iter = []

for exp in expansions:
    segs, widths, leaves = render_segments(exp, angle, base_length, shrink)
    segments_per_iter.append(segs)
    widths_per_iter.append(widths)
    leaves_per_iter.append(leaves)

# --- 描画 ---
fig, ax = plt.subplots(figsize=(6,8))
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-100,100)
ax.set_ylim(0,500)

lc = LineCollection([], colors="saddlebrown")
ax.add_collection(lc)
leaf_scat = ax.scatter([], [], color="green")

def update(frame):
    # 枝累積描画
    segs_list = [segments_per_iter[i] for i in range(frame+1) if len(segments_per_iter[i])>0]
    widths_list = [widths_per_iter[i] for i in range(frame+1) if len(widths_per_iter[i])>0]
    if segs_list:
        lc.set_segments(np.vstack(segs_list))
        lc.set_linewidth(np.hstack(widths_list))
    else:
        lc.set_segments(np.empty((0,2,2)))

    # 葉累積描画
    all_leaves = [leaf for i in range(frame+1) for leaf in leaves_per_iter[i]]
    if all_leaves:
        lx, ly, sizes = zip(*all_leaves)
        leaf_scat.set_offsets(np.column_stack([lx, ly]))
        leaf_scat.set_sizes(sizes)
    else:
        leaf_scat.set_offsets(np.empty((0,2)))
        leaf_scat.set_sizes([])

    ax.set_title("A growing plant")
    return lc, leaf_scat

ani = animation.FuncAnimation(fig, update, frames=max_iter+1, interval=800, blit=True, repeat=False)
plt.show()
