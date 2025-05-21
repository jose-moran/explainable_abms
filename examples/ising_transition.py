import numpy as np
import matplotlib.pyplot as plt


# Define the function for the mean-field self-consistency equation
def tanh_curve(m_vals, J):
    return np.tanh(J * m_vals)


# Use qualitative values
J_values = [0.5, 1.5]
colors = ["blue", "red"]

# Setup plot
fig, ax = plt.subplots(figsize=(8, 6))
m = np.linspace(-1.0, 1.0, 500)

# Plot identity line
ax.plot(m, m, "k--")

# Plot tanh(Jm) for different J
for J, color in zip(J_values, colors):
    ax.plot(m, tanh_curve(m, J), color=color)

# Indicate slopes at m = 0
for J, color in zip(J_values, colors):
    slope = J
    ax.plot([0, 0.2], [0, slope * 0.2], color=color, linestyle=":", linewidth=2)

# Mark fixed points (intercepts)
for J, color in zip(J_values, colors):
    m_vals = np.linspace(-1, 1, 1000)
    diff = m_vals - np.tanh(J * m_vals)
    zero_crossings = np.where(np.diff(np.sign(diff)))[0]
    for idx in zero_crossings:
        m_star = m_vals[idx]
        ax.plot(m_star, m_star, "o", color=color)

# Final formatting
ax.set_title("Self-Consistency in Mean-Field Ising Model")
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_position("zero")
ax.spines["left"].set_position("zero")

plt.tight_layout()
plt.show()
