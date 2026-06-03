#basic implementation of schelling model. w/ cost function

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

L = 50
THRESHOLD = 0.4
p_A = 0.4
p_B = 0.4
p_empty = 1 - p_A - p_B


def make_grid(l=L):
    return np.random.choice([1, 2, 0], size=(l, l), p=[p_A, p_B, p_empty])


def get_neighbors(grid, x, y):
    l = grid.shape[0]
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx = (x + dx) % l
            ny = (y + dy) % l
            neighbors.append(grid[nx, ny])
    return neighbors


def is_happy(grid, x, y):
    agent_type = grid[x, y]
    if agent_type == 0:
        return True
    neighbors = get_neighbors(grid, x, y)
    occupied = [n for n in neighbors if n != 0]
    if len(occupied) == 0:
        return True
    same_type = sum(1 for n in occupied if n == agent_type)
    return (same_type / len(occupied)) >= THRESHOLD


def mean_similarity_ratio(grid):
    l = grid.shape[0]
    ratios = []
    for x in range(l):
        for y in range(l):
            agent_type = grid[x, y]
            if agent_type == 0:
                continue
            neighbors = get_neighbors(grid, x, y)
            occupied = [n for n in neighbors if n != 0]
            if len(occupied) == 0:
                continue
            same_type = sum(1 for n in occupied if n == agent_type)
            ratios.append(same_type / len(occupied))
    return np.mean(ratios) if ratios else 0.0


def run_round(grid):
    l = grid.shape[0]
    unhappy = [
        (x, y)
        for x in range(l)
        for y in range(l)
        if grid[x, y] != 0 and not is_happy(grid, x, y)
    ]
    empty_spots = [
        (x, y)
        for x in range(l)
        for y in range(l)
        if grid[x, y] == 0
    ]

    random.shuffle(unhappy)
    random.shuffle(empty_spots)

    moves = 0
    for i, (x, y) in enumerate(unhappy):
        if i >= len(empty_spots):
            break
        ex, ey = empty_spots[i]
        grid[ex, ey] = grid[x, y]
        grid[x, y] = 0
        moves += 1

    return moves


def count_happy(grid):
    l = grid.shape[0]
    happy = 0
    unhappy = 0
    for x in range(l):
        for y in range(l):
            if grid[x, y] != 0:
                if is_happy(grid, x, y):
                    happy += 1
                else:
                    unhappy += 1
    return happy, unhappy


def cost(
    grid: np.ndarray,
    alpha: float = 0.5,
    msr_target: float = 0.5,
) -> float:
    """
        C =
        alpha * (1 - frac_happy)^2                  , individual metric (low when more agents happy)
         +
        (1 - alpha) * (MSR - msr_target)^2          , collective metric (low when MSR close to target)
    """
    l = grid.shape[0]
    happy = 0
    unhappy = 0
    ratios = []

    for x in range(l):
        for y in range(l):
            agent = grid[x, y]
            if agent == 0:
                continue
            neighbors = get_neighbors(grid, x, y)
            occupied = [n for n in neighbors if n != 0]
            if len(occupied) == 0:
                happy += 1
                continue
            same = sum(1 for n in occupied if n == agent)
            frac_same = same / len(occupied)
            ratios.append(frac_same)
            if frac_same >= THRESHOLD:
                happy += 1
            else:
                unhappy += 1

    total_occupied = happy + unhappy
    frac_happy = (happy / total_occupied) if total_occupied > 0 else 0.0
    msr = float(np.mean(ratios)) if ratios else 0.0

    return alpha * (1.0 - frac_happy) ** 2 + (1.0 - alpha) * (msr - msr_target) ** 2


def generate_history(grid, rounds=20):
    history = []
    stats = []

    for _ in range(rounds):
        history.append(grid.copy())
        happy, unhappy = count_happy(grid)
        msr = mean_similarity_ratio(grid)
        moves = run_round(grid)
        stats.append((happy, unhappy, moves, msr))

    history.append(grid.copy())
    happy, unhappy = count_happy(grid)
    msr = mean_similarity_ratio(grid)
    stats.append((happy, unhappy, 0, msr))

    return history, stats


def make_movie(grid, rounds=20):
    history, stats = generate_history(grid, rounds)

    cmap = ListedColormap(["white", "blue", "red"])
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        im.set_data(history[frame])
        happy, unhappy, moves, msr = stats[frame]
        ax.set_title(
            f"Round {frame}/{rounds} | Happy={happy} Unhappy={unhappy} | Moves={moves} | MSR={msr:.3f}"
        )
        return [im]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=50,
        repeat=False,
    )

    plt.show()
    return anim


if __name__ == "__main__":
    grid = make_grid()
    make_movie(grid, rounds=100)
    print("Final cost:", cost(grid, alpha=0.5, msr_target=0.5))
