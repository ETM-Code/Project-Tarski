import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Use a nice style
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['figure.facecolor'] = 'white'

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# X range
x = np.linspace(-2, 2, 1000)

# Heaviside step function
heaviside = np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))

# FastSigmoid surrogate gradient (slope=25)
slope_fast = 25
fast_sigmoid = 1 / (slope_fast * np.abs(x) + 1)**2

# ATan surrogate gradient (alpha=2)
alpha = 2
atan_grad = (alpha / 2) / (1 + (np.pi * alpha * x / 2)**2)

# Sigmoid surrogate gradient (slope=10 for visibility)
slope_sig = 10
sigmoid_grad = slope_sig * np.exp(-slope_sig * x) / (np.exp(-slope_sig * x) + 1)**2

# Colours matching the document theme
darkblue = '#003366'
orange = '#E67E22'
green = '#27AE60'
purple = '#8E44AD'

# Plot Heaviside
ax.plot(x[x < 0], heaviside[x < 0], color=darkblue, linewidth=2.5, label='Heaviside (forward)')
ax.plot(x[x > 0], heaviside[x > 0], color=darkblue, linewidth=2.5)
ax.scatter([0], [0.5], color=darkblue, s=80, zorder=5)

# Plot surrogate gradients
ax.plot(x, fast_sigmoid, color=orange, linewidth=2.5, label='FastSigmoid gradient')
ax.plot(x, atan_grad, color=green, linewidth=2.5, linestyle='--', label='ATan gradient')
ax.plot(x, sigmoid_grad, color=purple, linewidth=2.5, linestyle=':', label='Sigmoid gradient')

# Formatting
ax.set_xlabel(r'$V - \theta$ (distance from threshold)', fontsize=14)
ax.set_ylabel('Value / Gradient', fontsize=14)
ax.set_xlim(-2, 2)
ax.set_ylim(-0.05, 1.15)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')
ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='-')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

# Add annotation
ax.annotate('Surrogate gradients are\nnon-zero near threshold',
            xy=(0, 0.8), xytext=(0.8, 0.85),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

plt.tight_layout()
plt.savefig('surrogate_gradients.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('surrogate_gradients.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved surrogate_gradients.png and .pdf")
