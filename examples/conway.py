"""
Conway's Game of Life - A Classic Cellular Automaton

This module implements Conway's Game of Life, a cellular automaton that demonstrates
how complex patterns can emerge from simple rules. Each cell in a 2D grid can be either
alive (1) or dead (0), and its state in the next generation depends only on its current
state and the states of its eight immediate neighbors.

Rules:
1. A live cell with 2 or 3 live neighbors stays alive
2. A dead cell with exactly 3 live neighbors becomes alive
3. All other cells die or stay dead

This implementation uses NumPy for efficient grid operations and Matplotlib for
visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, List

# Grid size
N: int = 100

# Probability that a cell is alive at the start
INITIAL_DENSITY: float = 0.2

# Create initial grid
grid: np.ndarray = np.random.choice(
    [0, 1], size=(N, N), p=[1 - INITIAL_DENSITY, INITIAL_DENSITY]
)


def get_neighbors(grid: np.ndarray, i: int, j: int) -> List[int]:
    """
    Get the states of all eight neighbors of a cell, using periodic boundary conditions.

    Args:
        grid: The current state of the grid
        i: Row index of the cell
        j: Column index of the cell

    Returns:
        List of neighbor states (0 or 1) in the following order:
        [left, right, up, down, up-left, up-right, down-left, down-right]
    """
    return [
        grid[i, (j - 1) % N],  # left
        grid[i, (j + 1) % N],  # right
        grid[(i - 1) % N, j],  # up
        grid[(i + 1) % N, j],  # down
        grid[(i - 1) % N, (j - 1) % N],  # up-left
        grid[(i - 1) % N, (j + 1) % N],  # up-right
        grid[(i + 1) % N, (j - 1) % N],  # down-left
        grid[(i + 1) % N, (j + 1) % N],  # down-right
    ]


def update(
    frame_num: int, img: plt.AxesImage, grid: np.ndarray
) -> Tuple[plt.AxesImage, ...]:
    """
    Update the grid according to Conway's Game of Life rules.

    Args:
        frame_num: The current frame number (unused, required by animation)
        img: The matplotlib image object to update
        grid: The current state of the grid

    Returns:
        Tuple containing the updated image object
    """
    new_grid = grid.copy()
    for i in range(N):
        for j in range(N):
            # Count live neighbors
            neighbors = get_neighbors(grid, i, j)
            total = sum(neighbors)

            # Apply Conway's rules:
            # 1. Live cell with 2 or 3 neighbors survives
            # 2. Dead cell with exactly 3 neighbors becomes alive
            # 3. All other cells die or stay dead
            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    new_grid[i, j] = 0
            else:
                if total == 3:
                    new_grid[i, j] = 1
    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return (img,)


def main() -> None:
    """
    Create and display an animated visualization of Conway's Game of Life.
    The animation runs for 200 frames with a 100ms interval between frames.
    """
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation="nearest", cmap="binary")
    ani = animation.FuncAnimation(
        fig, update, fargs=(img, grid), frames=200, interval=100, save_count=50
    )
    plt.axis("off")
    plt.title("Conway's Game of Life")
    plt.show()


if __name__ == "__main__":
    main()
