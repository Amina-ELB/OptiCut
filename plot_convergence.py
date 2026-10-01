import matplotlib.pyplot as plt
import numpy as np

cost = np.loadtxt('doc/images/demo_compliance_3D/cost_func.txt')
constraint = np.loadtxt('doc/images/demo_compliance_3D/constraint.txt')

fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Iterations', fontsize=12)
ax1.set_ylabel('Compliance', color=color, fontsize=12)
ax1.plot(cost, color=color, linewidth=2, label='Compliance')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Constraint C(Ω)', color=color, fontsize=12)
ax2.plot(constraint, color=color, linewidth=2, linestyle='--', label='Constraint')
ax2.axhline(y=0.0, color='black', linestyle=':', label='Target Constraint (0)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Evolution of the Objective Function and Constraint', fontsize=14, pad=15)
fig.tight_layout()

plt.savefig('doc/images/demo_compliance_3D/convergence.png', dpi=300, bbox_inches='tight')
