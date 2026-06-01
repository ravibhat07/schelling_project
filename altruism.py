import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap

L = 50
p_A = 0.38
p_B = 0.38
p_altruist = 0.04
p_empty = 1 - p_A - p_B - p_altruist

grid = np.random.choice(
    [1, 2, 3, 0],
    size=(L, L),
    p=[p_A, p_B, p_altruist, p_empty],
)

_OFFSETS = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if dx or dy]


def _nbr_vals(x, y):
    return [grid[(x + dx) % L, (y + dy) % L] for dx, dy in _OFFSETS]


def _nbr_coords(x, y):
    return [((x + dx) % L, (y + dy) % L) for dx, dy in _OFFSETS]


def is_happy(x, y):
    t = grid[x, y]
    if t == 0:
        return True
    nbrs = _nbr_vals(x, y)
    if t == 3:
        return False  # altruists always try to find a better position
    occ = [n for n in nbrs if n != 0]
    if not occ:
        return True
    return sum(1 for n in occ if n == t) / len(occ) >= 0.3


def _egoist_stats(x, y):
    """(is_happy: bool, ratio: float) for egoist at (x, y), else (None, None)."""
    t = grid[x, y]
    if t == 0 or t == 3:
        return None, None
    nbrs = _nbr_vals(x, y)
    occ = [n for n in nbrs if n != 0]
    if not occ:
        return True, 1.0
    r = sum(1 for n in occ if n == t) / len(occ)
    return r >= 0.3, r


def _cost_from_parts(happy, unhappy, sum_r, num_r, alpha, msr_target):
    total = happy + unhappy
    fh = happy / total if total else 0.0
    msr = sum_r / num_r if num_r else 0.0
    return alpha * (1 - fh) ** 2 + (1 - alpha) * (msr - msr_target) ** 2


class _CostState:
    """
    Incrementally tracks cost() components so altruist candidate moves
    pay O(16 cells) instead of O(L²) per evaluation.
    """

    def __init__(self, alpha=0.5, threshold=0.3, msr_target=0.5):
        self.alpha = alpha
        self.threshold = threshold
        self.msr_target = msr_target
        self.happy = self.unhappy = self.num_r = 0
        self.sum_r = 0.0

    def build(self):
        self.happy = self.unhappy = self.num_r = 0
        self.sum_r = 0.0
        for x in range(L):
            for y in range(L):
                h, r = _egoist_stats(x, y)
                if h is None:
                    continue
                if h:
                    self.happy += 1
                else:
                    self.unhappy += 1
                self.sum_r += r
                self.num_r += 1

    def value(self):
        return _cost_from_parts(
            self.happy, self.unhappy, self.sum_r, self.num_r,
            self.alpha, self.msr_target,
        )

    def _local_delta(self, sx, sy, dx, dy, agent_type):
        """
        Compute (dh, du, dsum_r, dnum_r) for moving agent_type (sx,sy)→(dx,dy).
        Temporarily modifies grid, then reverts.
        """
        affected = set(_nbr_coords(sx, sy)) | set(_nbr_coords(dx, dy))
        affected.add((sx, sy))
        affected.add((dx, dy))

        before = {cell: _egoist_stats(*cell) for cell in affected}
        grid[dx, dy] = agent_type
        grid[sx, sy] = 0
        after = {cell: _egoist_stats(*cell) for cell in affected}
        grid[sx, sy] = agent_type
        grid[dx, dy] = 0

        dh = du = dnum_r = 0
        dsum_r = 0.0
        for cell in affected:
            h0, r0 = before[cell]
            h1, r1 = after[cell]
            if h0 is not None:
                if h0: dh -= 1
                else: du -= 1
                dsum_r -= r0
                dnum_r -= 1
            if h1 is not None:
                if h1: dh += 1
                else: du += 1
                dsum_r += r1
                dnum_r += 1
        return dh, du, dsum_r, dnum_r

    def hypothetical(self, sx, sy, dx, dy, agent_type):
        """Cost if agent_type moves (sx,sy)→(dx,dy), without committing."""
        dh, du, dsr, dnr = self._local_delta(sx, sy, dx, dy, agent_type)
        return _cost_from_parts(
            self.happy + dh, self.unhappy + du,
            self.sum_r + dsr, self.num_r + dnr,
            self.alpha, self.msr_target,
        )

    def commit(self, sx, sy, dx, dy, agent_type):
        """Apply move to grid and update state incrementally."""
        affected = set(_nbr_coords(sx, sy)) | set(_nbr_coords(dx, dy))
        affected.add((sx, sy))
        affected.add((dx, dy))

        before = {cell: _egoist_stats(*cell) for cell in affected}
        grid[dx, dy] = agent_type
        grid[sx, sy] = 0
        after = {cell: _egoist_stats(*cell) for cell in affected}

        for cell in affected:
            h0, r0 = before[cell]
            h1, r1 = after[cell]
            if h0 is not None:
                if h0: self.happy -= 1
                else: self.unhappy -= 1
                self.sum_r -= r0
                self.num_r -= 1
            if h1 is not None:
                if h1: self.happy += 1
                else: self.unhappy += 1
                self.sum_r += r1
                self.num_r += 1


_cost_state = _CostState()



def count_happy_and_msr():
    happy = unhappy = 0
    ratios = []
    for x in range(L):
        for y in range(L):
            t = grid[x, y]
            if t == 0:
                continue
            if is_happy(x, y):
                happy += 1
            else:
                unhappy += 1
            if t != 3:
                nbrs = _nbr_vals(x, y)
                occ = [n for n in nbrs if n != 0]
                if occ:
                    ratios.append(sum(1 for n in occ if n == t) / len(occ))
    msr = float(np.mean(ratios)) if ratios else 0.0
    return happy, unhappy, msr


def count_happy():
    h, u, _ = count_happy_and_msr()
    return h, u


def mean_similarity_ratio():
    _, _, msr = count_happy_and_msr()
    return msr



def run_round():
    unhappy = []
    for x in range(L):
        for y in range(L):
            if grid[x, y] != 0 and not is_happy(x, y):
                unhappy.append((x, y))

    # np.argwhere is faster than a nested Python loop for large grids
    empty_spots = list(map(tuple, np.argwhere(grid == 0)))

    random.shuffle(unhappy)
    random.shuffle(empty_spots)

    # Build cost state once if any unhappy altruists exist; maintain it
    # incrementally so each candidate evaluation is O(16) not O(L²).
    altruist_present = any(grid[x, y] == 3 for x, y in unhappy)
    if altruist_present:
        _cost_state.build()

    moves = 0
    empty_idx = 0

    for ax, ay in unhappy:
        if empty_idx >= len(empty_spots):
            break
        agent_type = grid[ax, ay]
        if agent_type == 0:
            continue  # safety: shouldn't happen given disjoint sets

        if agent_type == 3:
            current_cost = _cost_state.value()
            best_cost = current_cost
            best_i = None
            for i in range(empty_idx, min(empty_idx + 20, len(empty_spots))):
                ex, ey = empty_spots[i]
                c = _cost_state.hypothetical(ax, ay, ex, ey, agent_type)
                if c < best_cost:
                    best_cost = c
                    best_i = i
            if best_i is None:
                continue  # no beneficial move found; stay put
            target_i = best_i
        else:
            target_i = empty_idx

        ex, ey = empty_spots[target_i]
        if altruist_present:
            # commit() applies the move to grid and keeps state consistent
            _cost_state.commit(ax, ay, ex, ey, agent_type)
        else:
            grid[ex, ey] = agent_type
            grid[ax, ay] = 0

        empty_spots[target_i] = (ax, ay)
        moves += 1
        empty_idx += 1

    return moves



def cost(g=None, alpha=0.5, threshold=0.3, msr_target=0.5):
    if g is None:
        g = grid
    L_local = g.shape[0]
    happy = unhappy = 0
    ratios = []
    for x in range(L_local):
        for y in range(L_local):
            t = g[x, y]
            if t == 0 or t == 3:
                continue
            nbrs = [g[(x + dx) % L_local, (y + dy) % L_local] for dx, dy in _OFFSETS]
            occ = [n for n in nbrs if n != 0]
            if not occ:
                happy += 1
                ratios.append(1.0)
                continue
            same = sum(1 for n in occ if n == t)
            r = same / len(occ)
            ratios.append(r)
            if r >= threshold:
                happy += 1
            else:
                unhappy += 1
    total = happy + unhappy
    fh = happy / total if total else 0.0
    msr = float(np.mean(ratios)) if ratios else 0.0
    return alpha * (1 - fh) ** 2 + (1 - alpha) * (msr - msr_target) ** 2



def generate_history(rounds=20):
    history = []
    stats = []
    for _ in range(rounds):
        history.append(grid.copy())
        happy, unhappy, msr = count_happy_and_msr()
        moves = run_round()
        stats.append((happy, unhappy, moves, msr))
    history.append(grid.copy())
    happy, unhappy, msr = count_happy_and_msr()
    stats.append((happy, unhappy, 0, msr))
    return history, stats


def make_movie(rounds=20):
    history, stats = generate_history(rounds)

    cmap = ListedColormap(["white", "blue", "red", "yellow"])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(history[0], cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks([])
    ax.set_yticks([])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="blue",   label="Egoist A"),
        Patch(facecolor="red",    label="Egoist B"),
        Patch(facecolor="yellow", label="Altruist"),
        Patch(facecolor="white",  edgecolor="grey", label="Empty"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    def update(frame):
        im.set_data(history[frame])
        happy, unhappy, moves, msr = stats[frame]
        ax.set_title(
            f"Round {frame}/{rounds} | Happy={happy} Unhappy={unhappy} "
            f"| Moves={moves} | MSR={msr:.3f}"
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


make_movie(rounds=100)
print("Final cost:", cost(grid, alpha=0.5, threshold=0.3, msr_target=0.5))
