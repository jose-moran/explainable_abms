"""
Flocking Behavior Comparison: Topological vs. Metric Neighbors

This module implements and compares two flocking models based on the work of Bialek et al. (2011):
"Statistical mechanics for natural flocks of birds" [arXiv:1107.0604]

The paper demonstrated that starling flocks use topological interactions (k nearest neighbors)
rather than metric interactions (fixed radius), which inspired this comparison.

This module implements and compares two flocking models:
1. Topological neighbors: Birds interact with their k nearest neighbors
2. Metric neighbors: Birds interact with all neighbors within a fixed radius

Both models include:
- Alignment: Birds tend to align their velocities with neighbors
- Cohesion: Birds are attracted to the local center of mass
- Noise: Random perturbations to prevent complete synchronization
- Predator avoidance: Birds are repelled by a wandering predator

The simulation visualizes both models side-by-side to compare their emergent behaviors.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial import KDTree
from typing import Tuple, List, Optional, Any


# --- Predator class ---
class Predator:
    """
    A wandering predator that disturbs the flock.

    The predator moves with inertia and random direction changes,
    creating a repulsive force on nearby birds.
    """

    def __init__(self, L: float, speed: float = 0.07) -> None:
        """
        Initialize predator with random position and direction.

        Args:
            L: Size of the square domain
            speed: Movement speed of the predator
        """
        self.L = L
        self.position = np.random.rand(2) * L
        angle = np.random.rand() * 2 * np.pi
        self.velocity = np.array([np.cos(angle), np.sin(angle)])
        self.speed = speed

    def step(self) -> None:
        """
        Update predator position with random direction changes.
        Uses periodic boundary conditions to keep predator in domain.
        """
        angle_change = np.random.uniform(-np.pi / 8, np.pi / 8)
        angle = np.arctan2(self.velocity[1], self.velocity[0]) + angle_change
        self.velocity = np.array([np.cos(angle), np.sin(angle)])
        self.position = (self.position + self.speed * self.velocity) % self.L

    def get_position(self) -> np.ndarray:
        """Return current position of the predator."""
        return self.position


# --- Repulsion from predator ---
def predator_repulsion(
    bird_pos: np.ndarray, predator_pos: np.ndarray, radius: float, strength: float
) -> np.ndarray:
    """
    Calculate repulsive force from predator to bird.

    Args:
        bird_pos: Position of the bird
        predator_pos: Position of the predator
        radius: Maximum distance for repulsion
        strength: Strength of repulsion force

    Returns:
        Repulsive force vector (zero if predator is too far)
    """
    vec = bird_pos - predator_pos
    dist = np.linalg.norm(vec)
    if dist < radius and dist > 1e-6:
        return strength * vec / dist
    return np.zeros(2)


# --- Topological Flocking with predator ---
class FlockSimulationTopological:
    """
    Flocking simulation using topological neighbors (k nearest neighbors).
    Birds interact with their k closest neighbors regardless of distance.
    """

    def __init__(
        self,
        predator: Predator,
        N: int = 100,
        L: float = 10.0,
        nc: int = 7,
        alpha: float = 0.5,
        beta: float = 0.3,
        eta: float = 0.1,
        speed: float = 0.05,
        rep_radius: float = 1.0,
        rep_strength: float = 1.5,
    ) -> None:
        """
        Initialize flock with random positions and velocities.

        Args:
            predator: The predator object
            N: Number of birds
            L: Size of square domain
            nc: Number of topological neighbors
            alpha: Alignment strength
            beta: Cohesion strength
            eta: Noise strength
            speed: Bird movement speed
            rep_radius: Predator repulsion radius
            rep_strength: Predator repulsion strength
        """
        self.predator = predator
        self.N = N
        self.L = L
        self.nc = nc
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.speed = speed
        self.rep_radius = rep_radius
        self.rep_strength = rep_strength
        self.positions = np.random.rand(N, 2) * L
        angles = np.random.rand(N) * 2 * np.pi
        self.velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def step(self) -> None:
        """
        Update bird positions and velocities for one time step.
        Uses KDTree for efficient neighbor finding.
        """
        tree = KDTree(self.positions)
        new_vel = np.zeros_like(self.velocities)
        predator_pos = self.predator.get_position()
        for i in range(self.N):
            _, idxs = tree.query(self.positions[i], k=self.nc + 1)
            neighbors = idxs[1:]  # Exclude self
            alignment = np.mean(self.velocities[neighbors], axis=0)
            cohesion = np.mean(self.positions[neighbors], axis=0) - self.positions[i]
            repulsion = predator_repulsion(
                self.positions[i], predator_pos, self.rep_radius, self.rep_strength
            )
            noise = np.array(
                [np.cos(theta := np.random.rand() * 2 * np.pi), np.sin(theta)]
            )
            total = (
                self.alpha * alignment
                + self.beta * cohesion
                + repulsion
                + self.eta * noise
            )
            new_vel[i] = total / np.linalg.norm(total)
        self.velocities = new_vel
        self.positions = (self.positions + self.speed * self.velocities) % self.L

    def get_positions(self) -> np.ndarray:
        """Return current positions of all birds."""
        return self.positions

    def get_velocities(self) -> np.ndarray:
        """Return current velocities of all birds."""
        return self.velocities


# --- Metric Flocking with predator ---
class FlockSimulationMetric:
    """
    Flocking simulation using metric neighbors (within fixed radius).
    Birds interact with all neighbors within a specified distance.
    """

    def __init__(
        self,
        predator: Predator,
        N: int = 100,
        L: float = 10.0,
        r: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.3,
        eta: float = 0.1,
        speed: float = 0.05,
        rep_radius: float = 1.0,
        rep_strength: float = 1.5,
    ) -> None:
        """
        Initialize flock with random positions and velocities.

        Args:
            predator: The predator object
            N: Number of birds
            L: Size of square domain
            r: Interaction radius for metric neighbors
            alpha: Alignment strength
            beta: Cohesion strength
            eta: Noise strength
            speed: Bird movement speed
            rep_radius: Predator repulsion radius
            rep_strength: Predator repulsion strength
        """
        self.predator = predator
        self.N = N
        self.L = L
        self.r = r
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.speed = speed
        self.rep_radius = rep_radius
        self.rep_strength = rep_strength
        self.positions = np.random.rand(N, 2) * L
        angles = np.random.rand(N) * 2 * np.pi
        self.velocities = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def step(self) -> None:
        """
        Update bird positions and velocities for one time step.
        Uses KDTree for efficient neighbor finding within radius.
        """
        tree = KDTree(self.positions)
        new_vel = np.zeros_like(self.velocities)
        predator_pos = self.predator.get_position()
        for i in range(self.N):
            idxs = tree.query_ball_point(self.positions[i], self.r)
            if i in idxs:
                idxs.remove(i)
            if not idxs:
                new_vel[i] = self.velocities[i]
                continue
            alignment = np.mean(self.velocities[idxs], axis=0)
            cohesion = np.mean(self.positions[idxs], axis=0) - self.positions[i]
            repulsion = predator_repulsion(
                self.positions[i], predator_pos, self.rep_radius, self.rep_strength
            )
            noise = np.array(
                [np.cos(theta := np.random.rand() * 2 * np.pi), np.sin(theta)]
            )
            total = (
                self.alpha * alignment
                + self.beta * cohesion
                + repulsion
                + self.eta * noise
            )
            new_vel[i] = total / np.linalg.norm(total)
        self.velocities = new_vel
        self.positions = (self.positions + self.speed * self.velocities) % self.L

    def get_positions(self) -> np.ndarray:
        """Return current positions of all birds."""
        return self.positions

    def get_velocities(self) -> np.ndarray:
        """Return current velocities of all birds."""
        return self.velocities


# --- Main function with animation ---
def main() -> None:
    """
    Create and run the side-by-side animation of both flocking models.
    Shows topological vs. metric neighbor interactions in real-time.
    """
    L = 10.0
    predator = Predator(L)
    sim_topo = FlockSimulationTopological(predator, L=L)
    sim_metric = FlockSimulationMetric(predator, L=L)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Topological Neighbors")
    ax2.set_title("Metric Neighbors")
    for ax in (ax1, ax2):
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal")
        ax.axis("off")

    pos1 = sim_topo.get_positions()
    vel1 = sim_topo.get_velocities()
    pos2 = sim_metric.get_positions()
    vel2 = sim_metric.get_velocities()

    scat1 = ax1.quiver(pos1[:, 0], pos1[:, 1], vel1[:, 0], vel1[:, 1], color="blue")
    scat2 = ax2.quiver(pos2[:, 0], pos2[:, 1], vel2[:, 0], vel2[:, 1], color="red")
    (pred_dot1,) = ax1.plot([], [], "ko", markersize=6)
    (pred_dot2,) = ax2.plot([], [], "ko", markersize=6)

    def update(frame: int) -> Tuple[Any, ...]:
        """
        Update animation frame by advancing both simulations.

        Args:
            frame: Current frame number (unused, required by animation)

        Returns:
            Tuple of updated plot elements
        """
        predator.step()
        sim_topo.step()
        sim_metric.step()
        pos1, vel1 = sim_topo.get_positions(), sim_topo.get_velocities()
        pos2, vel2 = sim_metric.get_positions(), sim_metric.get_velocities()
        ppos = predator.get_position()
        scat1.set_offsets(pos1)
        scat1.set_UVC(vel1[:, 0], vel1[:, 1])
        scat2.set_offsets(pos2)
        scat2.set_UVC(vel2[:, 0], vel2[:, 1])
        pred_dot1.set_data([ppos[0]], [ppos[1]])
        pred_dot2.set_data([ppos[0]], [ppos[1]])
        return (scat1, scat2, pred_dot1, pred_dot2)

    ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
