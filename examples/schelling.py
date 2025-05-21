import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
EMPTY = 0
RED = 1
BLUE = 2


class SchellingModel:
    def __init__(
        self, size: int, density: float, tolerance: float, red_ratio: float = 0.5
    ):
        self.size = size
        self.tolerance = tolerance
        self.grid = self.initialize_grid(size, density, red_ratio)

    def initialize_grid(
        self, size: int, density: float, red_ratio: float
    ) -> np.ndarray:
        total_cells = size * size
        num_agents = int(total_cells * density)
        num_red = int(num_agents * red_ratio)
        num_blue = num_agents - num_red

        cells = (
            [RED] * num_red + [BLUE] * num_blue + [EMPTY] * (total_cells - num_agents)
        )
        np.random.shuffle(cells)
        return np.array(cells).reshape((size, size))

    def is_happy(self, x: int, y: int) -> bool:
        agent = self.grid[x, y]
        if agent == EMPTY:
            return True

        neighbors = self.get_neighbors(x, y)
        if not neighbors:
            return True

        same_type = sum(1 for n in neighbors if n == agent)
        return (same_type / len(neighbors)) >= self.tolerance

    def get_neighbors(self, x: int, y: int):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = (x + dx) % self.size, (y + dy) % self.size
                val = self.grid[nx, ny]
                if val != EMPTY:
                    neighbors.append(val)
        return neighbors

    def step(self):
        unhappy_agents = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if self.grid[x, y] != EMPTY and not self.is_happy(x, y)
        ]
        empty_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if self.grid[x, y] == EMPTY
        ]

        np.random.shuffle(unhappy_agents)
        np.random.shuffle(empty_cells)

        for (x_old, y_old), (x_new, y_new) in zip(unhappy_agents, empty_cells):
            self.grid[x_new, y_new] = self.grid[x_old, y_old]
            self.grid[x_old, y_old] = EMPTY

    def get_rgb_grid(self) -> np.ndarray:
        color_map = {
            EMPTY: [1, 1, 1],  # white
            RED: [1, 0, 0],  # red
            BLUE: [0, 0, 1],  # blue
        }
        rgb = np.zeros((self.size, self.size, 3))
        for val, color in color_map.items():
            mask = self.grid == val
            rgb[mask] = color
        return rgb


def main():
    # Parameters
    size = 50
    density = 0.9
    tolerance = 0.6
    model = SchellingModel(size=size, density=density, tolerance=tolerance)

    # Matplotlib animation
    fig, ax = plt.subplots()
    img = ax.imshow(model.get_rgb_grid(), interpolation="nearest")
    ax.axis("off")
    ax.set_title("Schelling Model of Segregation")

    def update(frame):
        model.step()
        img.set_data(model.get_rgb_grid())
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
