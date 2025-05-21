"""
2D Ising Model - A Classic Statistical Physics Simulation

This module implements the 2D Ising model, a fundamental model in statistical physics
that demonstrates phase transitions. Each site on a grid holds a binary spin value (+1 or -1),
and spins interact with their four nearest neighbors.

The simulation runs two grids in parallel:
1. One with coupling strength J1 < Jc (below critical coupling), which remains disordered
2. One with coupling strength J2 > Jc (above critical coupling), which becomes ordered

The Metropolis algorithm is used for updates:
- If energy change ΔE ≤ 0, flip is accepted
- If energy change ΔE > 0, flip is accepted with probability exp(-ΔE)

This implementation uses NumPy for efficient grid operations and Matplotlib for
visualization, with periodic boundary conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import AxesImage
from typing import Tuple, Union

# Simulation parameters
N: int = 100  # grid size
steps_per_frame: int = 1000  # how many updates per frame
beta: float = 1.0  # inverse temperature (fixed for both simulations)
J1: float = 0.3  # below critical coupling
J2: float = 1.0  # above critical coupling

# Initialize random spins
grid1: np.ndarray = np.random.choice([-1, 1], size=(N, N))
grid2: np.ndarray = np.copy(grid1)


def delta_cost(grid: np.ndarray, i: int, j: int, J: float) -> float:
    """
    Calculate the energy change that would result from flipping a spin.

    Args:
        grid: The current state of the grid
        i: Row index of the spin to flip
        j: Column index of the spin to flip
        J: Coupling strength

    Returns:
        The energy change (ΔE) that would result from flipping the spin
    """
    # Periodic boundary conditions
    s = grid[i, j]
    neighbors = (
        grid[(i + 1) % N, j]  # down
        + grid[(i - 1) % N, j]  # up
        + grid[i, (j + 1) % N]  # right
        + grid[i, (j - 1) % N]  # left
    )
    return 2 * J * s * neighbors


def update_grid(grid: np.ndarray, J: float) -> np.ndarray:
    """
    Perform one frame's worth of Metropolis updates on the grid.

    Args:
        grid: The current state of the grid
        J: Coupling strength

    Returns:
        The updated grid after performing steps_per_frame updates
    """
    for _ in range(steps_per_frame):
        i, j = np.random.randint(0, N, size=2)
        dE = delta_cost(grid, i, j, J)
        # Metropolis rule: accept if ΔE ≤ 0 or with probability exp(-ΔE)
        if dE <= 0 or np.random.rand() < np.exp(-dE):
            grid[i, j] *= -1
    return grid


def update(frame: int, img1: AxesImage, img2: AxesImage) -> Tuple[AxesImage, AxesImage]:
    """
    Update both grids and their visualizations for the animation.

    Args:
        frame: The current frame number (unused, required by animation)
        img1: The matplotlib image object for the first grid
        img2: The matplotlib image object for the second grid

    Returns:
        Tuple containing both updated image objects
    """
    global grid1, grid2
    grid1 = update_grid(grid1, J1)
    grid2 = update_grid(grid2, J2)
    img1.set_array(grid1)
    img2.set_array(grid2)
    return img1, img2


def main() -> None:
    """
    Create and display an animated visualization of two Ising model simulations
    running in parallel with different coupling strengths.
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    img1 = axs[0].imshow(grid1, cmap="Greys", vmin=-1, vmax=1)
    img2 = axs[1].imshow(grid2, cmap="Greys", vmin=-1, vmax=1)
    axs[0].set_title(f"J = {J1} < Jc (disordered)")
    axs[1].set_title(f"J = {J2} > Jc (ordered)")
    for ax in axs:
        ax.axis("off")

    ani = animation.FuncAnimation(
        fig, update, fargs=(img1, img2), interval=25, blit=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
