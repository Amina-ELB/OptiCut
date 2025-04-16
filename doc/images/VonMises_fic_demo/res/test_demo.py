import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter



plt.rcParams['text.usetex'] = True
import math
plt.rcParams["font.family"] = "Times New Roman"



# Création des données
with open("cost_func.txt", 'r') as fic_init:
    b = fic_init.readlines()
    cost= [float(a.replace('\n',"")) for a in b]
    fic_init.close()
with open("constraint.txt", 'r') as fic_init:
    b = fic_init.readlines()
    constraint= [float(a.replace('\n',"")) for a in b]
    fic_init.close()
        
y_cost = cost  # Axe des X
y_constraint = constraint
x = np.arange(start = 1, stop = (len(cost)+1))  # Fonction

# Création de la figure avec deux axes y
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()



# Définir la taille de la vidéo à la moitié de l'écran (en pouces)
screen_width_pixels = 4000
screen_height_pixels = 2000
dpi = 96  # DPI estimé
half_screen_width_inches = (screen_width_pixels / 2) / dpi
screen_height_inches = screen_height_pixels / dpi

# Réduire la largeur à la moitié de l'écran
fig.set_size_inches(half_screen_width_inches, screen_height_inches)





ax1.set_xlim(0, 1000)
ax1.set_ylim(0, 14)
ax2.set_ylim(-0.4, 0.6)

ax1.set_xlabel("Iteration, n", fontsize=42)
ax1.set_ylabel(r"Cost, $J\left(\Omega\right)$", color='r', fontsize=42)
ax2.set_ylabel(r"Constraint, $C\left(\Omega\right)$", color='b', fontsize=42)
ax1.tick_params(axis='both', labelsize=36)
ax2.tick_params(axis='y', labelsize=36)

line1, = ax1.plot(x, y_cost, 'r-', label=r'$J\left(\Omega\right)$')
line2, = ax2.plot(x, y_constraint, 'b-', label=r'$C\left(\Omega\right)$')
point1, = ax1.plot([x[0]], [y_cost[0]], 'ro',markersize=15)  # Point en mouvement pour y_cost
point2, = ax2.plot([x[0]], [y_constraint[0]], 'bo',markersize=15)  # Point en mouvement pour y_constraint

ax1.legend(loc='upper left', fontsize=36)
ax2.legend(loc='upper right', fontsize=36)

def init():
    point1.set_data([x[0]], [y_cost[0]])
    point2.set_data([x[0]], [y_constraint[0]])
    return point1, point2

def update(frame):
    point1.set_data([x[frame]], [y_cost[frame]])
    point2.set_data([x[frame]], [y_constraint[frame]])
    return point1, point2

ani = FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=100)

# Sauvegarde en MP4
writer = FFMpegWriter(fps=100, metadata={"title": "Animation"})
ani.save("animation.mp4", writer=writer)

plt.show()



