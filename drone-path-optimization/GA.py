import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# === Global Parameters ===
# =========================

FIELD_WIDTH = 20.0     # meters (x: -10..10)
FIELD_HEIGHT = 10.0    # meters (y: -5..5)
NODE_SPACING = 1.0     # meters between node centers
DRONE_SPEED = 0.7      # m/s (constant)
COVERAGE_RADIUS = 0.6  # meters footprint radius on ground
GRID_RESOLUTION = 0.05 # meters per cell (1 cm requirement)

START_NODE = (-9.5, -4.5)  # must be a valid node center

# GA params (you can safely crank POP_SIZE to 10000 for long runs)
POP_SIZE = 100000         # set to 10000 for full-scale runs
GENS = 100
PATH_LENGTH_MIN = 300
PATH_LENGTH_MAX = 1000
PATH_LENGTH = PATH_LENGTH_MAX
MUTATION_RATE = 0.10
ELITISM = POP_SIZE // 10           # keep this many best individuals each gen

# === Roulette-style Fitness Coefficients (mirrored from GA_telsiks_roulette.py) ===
# C1 = 0.0        # reward for covered squares
# C2 = 0.0       # penalty for uncovered squares
# C3 = 0.0        # penalty for not returning to start (note: with full_path closure this won't trigger)
# C4 = -100000000.0      # penalty for battery exhaustion (>100%)
# C5 = 0.0       # penalty per each redundant window
# C6 = 0.0         # reward for battery remaining
# C8 = 0.0        # reward proportional to mice removed
# C9 = 0.0         # bonus if all mice are removed

C1 = 100.0        # reward for covered squares
C2 = -20.0       # penalty for uncovered squares
C3 = -1.0        # penalty for not returning to start (note: with full_path closure this won't trigger)
C4 = -100000000.0      # penalty for battery exhaustion (>100%)
C5 = -0.60       # penalty per each redundant window
C6 = 0.0         # reward for battery remaining
C8 = 10.0        # reward proportional to mice removed
C9 = 0.0         # bonus if all mice are removed

C10 = 1000.0      # award if mouse moves toward edge (edge-distance metric increases)
C11 = -200.0      # penalty if mouse moves away from edge
C12 = 50000.0     # strong award when a mouse is removed

# Mice parameters
NUM_MICE = 16

# Mirror roulette's rats count logic
number_of_rats = NUM_MICE

# Output
BEST_JSON_PATH = "best_run.json"
PLOT_EVERY = 10  # gen intervals to plot best path

# =============================
# === Node/Grid Definitions ===
# =============================

X_MIN = -FIELD_WIDTH / 2.0
X_MAX = FIELD_WIDTH / 2.0
Y_MIN = -FIELD_HEIGHT / 2.0
Y_MAX = FIELD_HEIGHT / 2.0

def generate_nodes() -> List[Tuple[float, float]]:
    """Return list of all node centers at 1 m spacing, within field bounds."""
    xs = np.arange(X_MIN + 0.5, X_MAX, NODE_SPACING)  # centers at .5
    ys = np.arange(Y_MIN + 0.5, Y_MAX, NODE_SPACING)
    nodes = [(float(x), float(y)) for y in ys for x in xs]
    return nodes

ALL_NODES = set(generate_nodes())
NODE_TO_IDX = {node: i for i, node in enumerate(sorted(ALL_NODES))}
IDX_TO_NODE = {i: node for node, i in NODE_TO_IDX.items()}

def is_valid_node(node: Tuple[float, float]) -> bool:
    return node in ALL_NODES


# ===========================
# === Move Encoding/Logic ===
# ===========================

# Move codes: up=1, left=2, down=3, right=4
MOVE_DELTAS = {
    1: (0.0, +NODE_SPACING),   # up / north
    2: (-NODE_SPACING, 0.0),   # left / west
    3: (0.0, -NODE_SPACING),   # down / south
    4: (+NODE_SPACING, 0.0),   # right / east
}

# Precompute valid neighbor map for each node (move_code -> next_node)
VALID_NEIGHBORS: Dict[Tuple[float, float], Dict[int, Tuple[float, float]]] = {}
for node in ALL_NODES:
    neigh = {}
    for code, (dx, dy) in MOVE_DELTAS.items():
        nxt = (round(node[0] + dx, 6), round(node[1] + dy, 6))
        if is_valid_node(nxt):
            neigh[code] = nxt
    VALID_NEIGHBORS[node] = neigh

def apply_move(node: Tuple[float, float], move_code: int) -> Tuple[float, float]:
    dx, dy = MOVE_DELTAS[move_code]
    nxt = (round(node[0] + dx, 6), round(node[1] + dy, 6))
    # keep in nodes; if invalid, stay (we'll penalize attempted invalid)
    return nxt if is_valid_node(nxt) else node

def opposite_move(move_code: int) -> int:
    # up<->down, left<->right
    return {1:3, 3:1, 2:4, 4:2}[move_code]

def apply_move_with_retries(
    node: Tuple[float, float],
    desired_move: int,
    prev_node: Tuple[float, float] | None = None,
    max_tries: int = 20
) -> Tuple[Tuple[float, float], int, int, int]:
    """
    Try desired_move first. If invalid, pick a valid alternative.
    Prefer not to step back to prev_node (if avoidable).
    Returns:
        next_node,
        rerouted_flag (1 if desired was invalid),
        tries_used (>=1),
        used_move_code (the move actually taken)
    """
    neigh = VALID_NEIGHBORS.get(node, {})
    # 1) Desired move works
    if desired_move in neigh:
        return neigh[desired_move], 0, 1, desired_move

    # 2) Fixed priority order for alternatives (clockwise-ish around desired)
    priority = {
        1: [4, 2, 3],  # desired up -> try right, left, down
        2: [1, 3, 4],  # desired left -> try up, down, right
        3: [2, 4, 1],  # desired down -> try left, right, up
        4: [3, 1, 2],  # desired right -> try down, up, left
    }[desired_move]

    # First pass: avoid immediate backtrack to prev_node
    ordered: List[Tuple[int, Tuple[float, float]]] = []
    for m in priority:
        dst = neigh.get(m)
        if dst is None:
            continue
        if prev_node is not None and dst == prev_node:
            continue
        ordered.append((m, dst))

    # If nothing available without backtracking, allow backtrack as last resort
    if not ordered and prev_node is not None:
        for m in priority:
            dst = neigh.get(m)
            if dst is not None and dst == prev_node:
                ordered.append((m, dst))
                break

    # Take the first valid alternative deterministically
    if ordered:
        used_move, dst = ordered[0]
        # tries_used = 1 (desired) + index of chosen alternative + 1
        tries_used = min(1 + 1, max_tries)
        return dst, 1, tries_used, used_move

    # Fallback: truly boxed in (should not happen on this grid) -> stay
    return node, 1, 1, desired_move

# ========================
# === Coverage Bitmap  ===
# ========================

def get_coverage_counts_bitmap(path_xy: List[Tuple[float, float]]) -> np.ndarray:
    """
    path_xy are metric coordinates (x_m, y_m) visited by drone.
    Returns uint8 coverage counts per 1cm cell.
    """
    grid_width = int(FIELD_WIDTH / GRID_RESOLUTION)
    grid_height = int(FIELD_HEIGHT / GRID_RESOLUTION)
    coverage_map = np.zeros((grid_height, grid_width), dtype=np.uint8)

    radius_cells = int(COVERAGE_RADIUS / GRID_RESOLUTION)

    # Precompute circular mask once
    y_grid, x_grid = np.ogrid[-radius_cells:radius_cells+1, -radius_cells:radius_cells+1]
    circle_mask = (x_grid**2 + y_grid**2) <= (radius_cells**2)

    for (x_m, y_m) in path_xy:
        x_idx = int((x_m - X_MIN) / GRID_RESOLUTION)
        y_idx = int((y_m - Y_MIN) / GRID_RESOLUTION)

        x_start = max(x_idx - radius_cells, 0)
        y_start = max(y_idx - radius_cells, 0)
        x_end = min(x_idx + radius_cells + 1, coverage_map.shape[1])
        y_end = min(y_idx + radius_cells + 1, coverage_map.shape[0])

        mask_x_start = max(radius_cells - x_idx, 0)
        mask_y_start = max(radius_cells - y_idx, 0)
        mask_x_end = mask_x_start + (x_end - x_start)
        mask_y_end = mask_y_start + (y_end - y_start)

        coverage_map[y_start:y_end, x_start:x_end] += circle_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]

    return coverage_map


# ==================================
# === Battery Model (Excel file) ===
# ==================================

def load_avg_drain_rate_from_excel(xlsx_path: str = "Battery_Model_Stats.xlsx") -> float:
    """
    Returns average drain rate in %/s from Excel columns:
    'Avg_Drain_Rate [%/s]', 'Duration [s]', 'Battery_Used [%]'.
    If 'Avg_Drain_Rate [%/s]' exists, average that; else compute Battery_Used/Duration row-wise then average.
    Fallback default if file missing or invalid.
    """
    try:
        import pandas as pd
        df = pd.read_excel(xlsx_path)
        rate_col = None
        for c in df.columns:
            if str(c).strip().lower().startswith("avg_drain_rate"):
                rate_col = c
                break
        if rate_col is not None:
            rates = pd.to_numeric(df[rate_col], errors="coerce").dropna()
            if len(rates) > 0:
                return float(rates.mean())

        # else compute Battery_Used[%]/Duration[s]
        dur_col = next(c for c in df.columns if str(c).strip().lower().startswith("duration"))
        used_col = next(c for c in df.columns if str(c).strip().lower().startswith("battery_used"))
        durations = pd.to_numeric(df[dur_col], errors="coerce")
        used = pd.to_numeric(df[used_col], errors="coerce")
        valid = (~durations.isna()) & (~used.isna()) & (durations > 0)
        if valid.sum() > 0:
            rates = (used[valid] / durations[valid]).astype(float)
            return float(rates.mean())

        # If we got here, fall through to default
    except Exception as e:
        print(f"[Battery] Using fallback average drain rate (reason: {e})")

    # Fallback: a conservative average %/s; adjust as you measure real data
    return 0.08  # e.g., 0.08% per second


AVG_DRAIN_RATE_PCT_PER_S = load_avg_drain_rate_from_excel()

# =================
# === Mice Sims ===
# =================

@dataclass
class MiceState:
    positions: List[Tuple[float, float]]  # current positions
    history: List[List[Tuple[float, float]]]  # per-mouse path over time

def place_mice_uniformly(num_mice: int) -> MiceState:
    nodes_list = list(ALL_NODES)
    chosen = random.sample(nodes_list, num_mice)
    history = [[pos] for pos in chosen]
    return MiceState(positions=list(chosen), history=history)

def sample_mice_positions(num_mice: int) -> List[Tuple[float, float]]:
    """Return a list of unique node positions for this generation's shared mice layout."""
    return random.sample(list(ALL_NODES), num_mice)

def mouse_step_opposite(m_pos: Tuple[float, float], incoming_move: int) -> Tuple[float, float]:
    """
    Move the mouse one node AWAY from the incoming drone direction.
    The escape direction is the SAME as the drone's incoming move.
    Returns a candidate that may be outside the field; simulate_mice() handles 'OUT'.
    """

    dx, dy = MOVE_DELTAS[incoming_move]
    cand = (round(m_pos[0] + dx, 6), round(m_pos[1] + dy, 6))
    return cand

def simulate_mice(drone_path_nodes: List[Tuple[float, float]],
                  move_seq: List[int],
                  mice_init: MiceState) -> Tuple[MiceState, int, int]:
    """
    Shared mice logic:
      - If the drone arrives on a mouse node, that mouse steps one node in the opposite
        direction of the drone's incoming move.
      - If that step leaves the field, mark the mouse as 'OUT' (deterred) and keep it out
        for the rest of the flight. History records 'OUT' from that step onward.
    Returns:
      final mice state (with history), deterred count, and bump count.
    """
    mice = MiceState(positions=list(mice_init.positions), history=[h[:] for h in mice_init.history])
    deterred = 0
    bumps = 0

    for step_idx in range(1, len(drone_path_nodes)):
        drone_prev = drone_path_nodes[step_idx - 1]
        drone_curr = drone_path_nodes[step_idx]

        # If the drone didn't move (should be rare with your retry logic), just extend histories
        if drone_prev == drone_curr:
            for i, pos in enumerate(mice.positions):
                mice.history[i].append(pos)
            continue

        # Determine incoming move code by comparing nodes
        dx = round(drone_curr[0] - drone_prev[0], 6)
        dy = round(drone_curr[1] - drone_prev[1], 6)
        if dy > 0:
            in_code = 1
        elif dx < 0:
            in_code = 2
        elif dy < 0:
            in_code = 3
        else:
            in_code = 4

        new_positions: List[Tuple[float, float] | str] = []

        for i, pos in enumerate(mice.positions):
            # If this mouse is already OUT, keep it OUT
            if pos == "OUT":
                new_positions.append("OUT")
                mice.history[i].append("OUT")
                continue

            # If the drone arrived on this mouse's node, attempt opposite step
            if pos == drone_curr:
                bumps += 1
                nxt = mouse_step_opposite(pos, in_code)  # may be invalid
                if not is_valid_node(nxt):
                    deterred += 1
                    new_positions.append("OUT")
                    mice.history[i].append("OUT")
                else:
                    new_positions.append(nxt)
                    mice.history[i].append(nxt)
            else:
                # No interaction; mouse stays
                new_positions.append(pos)
                mice.history[i].append(pos)

        mice.positions = new_positions  # type: ignore[assignment]

    return mice, deterred, bumps


# =========================
# === Path Construction ===
# =========================

def decode_moves_to_nodes(start_node: Tuple[float, float], moves: List[int]
                         ) -> Tuple[List[Tuple[float, float]], int]:
    """
    Convert move sequence into node coordinates (including start).
    Returns:
    path: sequence of node positions,
    retry_attempts: sum of tries used per step.
    """
    path = [start_node]
    retry_attempts = 0
    prev = None
    curr = start_node

    for desired in moves:
        nxt, _rerouted_flag, tries_used, _used = apply_move_with_retries(
            curr, desired, prev_node=prev, max_tries=20
        )
        retry_attempts += tries_used
        path.append(nxt)
        prev, curr = curr, nxt

    return path, retry_attempts

# === Helpers to mirror roulette fitness ===
def world_to_idx(x: float, y: float) -> Tuple[int, int]:
    """Convert world coords (node centers) to integer grid indices used for redundancy windowing."""
    return (int(round(x - X_MIN - 0.5)), int(round(y - Y_MIN - 0.5)))

def edge_distance_metric(pos: Tuple[float, float]) -> float:
    """Distance to the nearest field edge (higher => closer to edge)."""
    x, y = pos
    return min(FIELD_WIDTH / 2.0 - abs(x), FIELD_HEIGHT / 2.0 - abs(y))

# Battery usage function identical in structure to roulette code
DRONE_WAIT_TIME = 0  # roulette sets 0; keep identical
DRAIN_RATE = AVG_DRAIN_RATE_PCT_PER_S  # mirror roulette's global used for idle drain (0 in practice)
def compute_battery_used(path: List[Tuple[float, float]], drain_rate: float) -> float:
    """Total battery used (%): flight + idle (idle is zero with DRONE_WAIT_TIME=0)."""
    total_distance = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    flight_time = total_distance / DRONE_SPEED
    idle_drain_rate = len(path) * DRONE_WAIT_TIME * DRAIN_RATE
    return flight_time * drain_rate + idle_drain_rate


# =====================
# === Fitness Score ===
# =====================

def fitness_of_moves(moves: List[int], mice_seed_positions: List[Tuple[float, float]]) -> Tuple[float, Dict]:

    """
    Compute fitness identical to GA_telsiks_roulette.py:
    - Close the path (start + path + start)
    - Coverage-based reward/penalty using raw cell counts
    - Sliding-window redundancy penalty (LAST_N_STEPS = 10)
    - Battery shaping: remaining reward C6 and hard overuse penalty C4
    - Strong mice shaping (C10/C11 edge metric; C12 for OUT; C8*removed; C9 if all removed)
    """
    # 1) Decode genome -> node path
    path_nodes, _retry_attempts = decode_moves_to_nodes(START_NODE, moves)

    # Build full_path exactly like roulette: [START] + path + [START]
    # Our path_nodes already includes START_NODE as first element; exclude that once to avoid duplication
    inner_path = path_nodes[1:]  # moves applied after start
    full_path = [START_NODE] + inner_path + [START_NODE]

    # 2) Coverage bitmap and counts
    coverage_counts = get_coverage_counts_bitmap(full_path)
    covered_once = int(np.sum(coverage_counts >= 1))
    uncovered_count = int(coverage_counts.size - covered_once)

    # 3) Sliding-window redundancy penalty over LAST_N_STEPS
    LAST_N_STEPS = 10
    cells = [world_to_idx(p[0], p[1]) for p in full_path]
    window_points = LAST_N_STEPS + 1
    redundant_windows = 0
    for i in range(0, max(0, len(cells) - window_points + 1)):
        sub = cells[i:i + window_points]
        # If any node repeats within the window, count this window as redundant
        if len(set(sub)) < len(sub):
            redundant_windows += 1
    redundant_coverage = redundant_windows

    # 4) Battery usage (percent)
    battery_used = compute_battery_used(full_path, AVG_DRAIN_RATE_PCT_PER_S)

    # 5) Mice simulation on the fixed layout; get final positions
    mice0 = MiceState(
        positions=list(mice_seed_positions),
        history=[[pos] for pos in mice_seed_positions],
    )

    mice_final, deterred, bumps = simulate_mice(full_path, moves, mice0)
    final_positions = mice_final.positions

    # 6) Assemble score (identical structure/coefficients)
    score = 0.0
    score += C1 * covered_once
    score += C2 * uncovered_count
    score += C5 * redundant_coverage
    score += C6 * (100.0 - battery_used)
    if full_path[-1] != START_NODE:
        score += C3
    if battery_used > 100.0:
        score += C4

    # Mouse-based shaping
    movement_score = 0.0
    mice_removed_count = 0
    for old_pos, new_pos in zip(mice_seed_positions, final_positions):
        if new_pos == "OUT":
            movement_score += C12
            mice_removed_count += 1
        else:
            old_m = edge_distance_metric(old_pos)
            new_m = edge_distance_metric(new_pos)
            if new_m > old_m:
                movement_score += C10
            elif new_m < old_m:
                movement_score += C11
    # removal terms
    score += C8 * mice_removed_count
    if mice_removed_count == number_of_rats:
        score += C9
    score += movement_score

    # Diagnostics (names aligned to new scheme)
    diag = dict(
        covered_cells=covered_once,
        uncovered_cells=uncovered_count,
        redundant_windows=redundant_coverage,
        batt_used_pct=battery_used,
        mice_deterred = deterred,
        mice_bumps = bumps,
        path_nodes=full_path,
        mice_history=mice_final.history,
    )
    return float(score), diag


# ============================
# === GA Core (Roulette)  ====
# ============================

def init_population(pop_size: int, path_len: int) -> np.ndarray:
    """
    Random integers in {1,2,3,4}
    """
    return np.random.randint(1, 5, size=(pop_size, path_len), dtype=np.uint8)

def evaluate_population_prefix(pop: np.ndarray,
                               mice_seed_positions: List[Tuple[float, float]],
                               lengths: np.ndarray
                              ) -> Tuple[np.ndarray, List[Dict]]:
    """
    Evaluate each individual using only the first lengths[i] genes.
    lengths: array of ints, len == len(pop).
    """
    scores = np.zeros((len(pop),), dtype=np.float64)
    diags: List[Dict] = []
    for i, indiv in enumerate(pop):
        L = int(lengths[i])
        s, d = fitness_of_moves(indiv[:L].tolist(), mice_seed_positions)
        d["used_length"] = L
        scores[i] = s
        diags.append(d)
    return scores, diags


def evaluate_population(pop: np.ndarray,
                        mice_seed_positions: List[Tuple[float, float]]
                       ) -> Tuple[np.ndarray, List[Dict]]:
    scores = np.zeros((len(pop),), dtype=np.float64)
    diags: List[Dict] = []
    for i, indiv in enumerate(pop):
        s, d = fitness_of_moves(indiv.tolist(), mice_seed_positions)
        scores[i] = s
        diags.append(d)
    return scores, diags

def roulette_select(pop: np.ndarray, scores: np.ndarray, num: int) -> np.ndarray:
    """
    Select 'num' individuals with probability proportional to shifted-positive fitness.
    """
    min_s = scores.min()
    shifted = scores - min_s + 1e-12
    probs = shifted / shifted.sum()
    idxs = np.random.choice(len(pop), size=num, p=probs)
    return pop[idxs]

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-point crossover at random cut (1..L-1)
    """
    L = len(parent1)
    if L < 2:
        return parent1.copy(), parent2.copy()
    cut = np.random.randint(1, L)
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)
    child1[:cut] = parent1[:cut]
    child1[cut:] = parent2[cut:]
    child2[:cut] = parent2[:cut]
    child2[cut:] = parent1[cut:]
    return child1, child2

def mutate(indiv: np.ndarray, rate: float) -> None:
    """
    With prob 'rate' per gene, replace with a random different move in {1,2,3,4}.
    """
    L = len(indiv)
    mask = np.random.rand(L) < rate
    for i in np.where(mask)[0]:
        old = int(indiv[i])
        # choose a different move
        choices = [1, 2, 3, 4]
        choices.remove(old)
        indiv[i] = np.random.choice(choices)

# ======================
# === Visualization  ===
# ======================

def plot_path(nodes: List[Tuple[float, float]], title: str = ""):
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    plt.figure()
    plt.plot(xs, ys, marker='o', linewidth=1)
    plt.scatter([xs[0]], [ys[0]], marker='s', s=60)  # start
    plt.title(title)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim([X_MIN, X_MAX])
    plt.ylim([Y_MIN, Y_MAX])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show(block=False)   # <<< non-blocking
    plt.pause(0.001)        # let UI update briefly
    plt.close()             # avoid accumulating windows


def plot_fitness(history: List[float]):
    plt.figure()
    plt.plot(history)
    plt.title("Best fitness per generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show(block=False)   # <<< non-blocking
    plt.pause(0.001)
    plt.close()

# ==========================
# === Save JSON Results  ===
# ==========================

def save_best_json(path_nodes: List[Tuple[float, float]],
                   mice_history: List[List[Tuple[float, float]]],
                   diag: Dict,
                   out_path: str = BEST_JSON_PATH):
    data = {
        "start_node": START_NODE,
        "best_path_nodes": path_nodes,
        "best_path_moves": None,  # to fill if needed by caller
        "mice_paths": mice_history,
        "diagnostics": {
            k: v for k, v in diag.items()
            if k not in ("path_nodes", "mice_history")
        }
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Saved] {out_path}")


# ===============
# === Runner ====
# ===============

def main():
    start_dt = datetime.now()
    print(f"[TIMING] Start: {start_dt:%Y-%m-%d %H:%M:%S}")
    print(f"Avg drain rate used: {AVG_DRAIN_RATE_PCT_PER_S:.5f} %/s")
    population = init_population(POP_SIZE, PATH_LENGTH)

    # Fixed mice layout for the entire run (same for all generations)
    fixed_mice_positions = sample_mice_positions(NUM_MICE)
    # fixed_mice_positions = [(sx * x, sy * y) for sx in (1, -1) for sy in (1, -1) for x in (2.5, 7.5) for y in (1.5, 3.5)]

    # Gen-1: random path lengths per individual in [PATH_LENGTH_MIN, PATH_LENGTH_MAX]
    gen1_lengths = np.random.randint(PATH_LENGTH_MIN, PATH_LENGTH_MAX + 1,
                                     size=POP_SIZE, dtype=int)

    ACTIVE_LENGTH = None  # will be decided after Gen 1

    best_history = []
    best_diag = None
    best_moves = None
    best_score = -1e18

    for gen in range(1, GENS + 1):
        # Use the same mice layout every generation
        # Gen 1: evaluate with random lengths (per individual)
        if gen == 1:
            scores, diags = evaluate_population_prefix(population, fixed_mice_positions, gen1_lengths)
        else:
            # From Gen 2 onward: evaluate with the fixed winning length
            assert ACTIVE_LENGTH is not None
            fixed_lengths = np.full(POP_SIZE, ACTIVE_LENGTH, dtype=int)
            scores, diags = evaluate_population_prefix(population, fixed_mice_positions, fixed_lengths)

        # Track best
        gen_best_idx = int(np.argmax(scores))
        gen_best_score = float(scores[gen_best_idx])
        gen_best_diag = diags[gen_best_idx]

        if gen == 1 and ACTIVE_LENGTH is None:
            # Take the winner’s length as the desired task length
            ACTIVE_LENGTH = int(gen_best_diag.get("used_length", PATH_LENGTH_MAX))
            # print(f"[LENGTH] Winning length selected from Gen 1: {ACTIVE_LENGTH}")

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_diag = gen_best_diag
            best_moves = population[gen_best_idx].copy()

        best_history.append(gen_best_score)

        print(f"Gen {gen:3d} | best={gen_best_score:.6f} | "
              f"covered_cells={gen_best_diag['covered_cells']} "
              f"uncovered={gen_best_diag['uncovered_cells']} "
              f"batt%={gen_best_diag['batt_used_pct']:.2f} "
              f"m_det={gen_best_diag['mice_deterred']}")

        # Plot current best path every PLOT_EVERY gens
        if gen % PLOT_EVERY == 0:
            plot_path(gen_best_diag["path_nodes"], title=f"Best path at gen {gen}")

        # === Next generation ===
        # Elitism
        elite_idx = np.argsort(scores)[-ELITISM:]  # top ELITISM
        elites = population[elite_idx].copy()

        # Roulette selection for the rest
        parents = roulette_select(population, scores, POP_SIZE - ELITISM)
        np.random.shuffle(parents)

        # Crossover (pairwise)
        children = []
        for i in range(0, len(parents) - 1, 2):
            c1, c2 = crossover(parents[i], parents[i+1])
            children.append(c1)
            children.append(c2)
        if len(children) < (POP_SIZE - ELITISM):
            # If odd, carry over one parent
            children.append(parents[-1].copy())
        children = np.array(children[:(POP_SIZE - ELITISM)], dtype=np.uint8)

        # Mutation
        for indiv in children:
            mutate(indiv, MUTATION_RATE)

        # New population
        population = np.vstack([elites, children])

    end_dt = datetime.now()
    elapsed = end_dt - start_dt
    print(f"[TIMING] End of last generation: {end_dt:%Y-%m-%d %H:%M:%S} | Elapsed: {elapsed}")

    # End GA
    assert best_diag is not None and best_moves is not None
    print("\n=== GA Finished ===")
    print(f"Best fitness: {best_score:.6f}")
    print(f"Covered cells: {best_diag['covered_cells']}, "
          f"Uncovered cells: {best_diag['uncovered_cells']}, "
          f"Battery %: {best_diag['batt_used_pct']:.2f}, "
          f"Mice deterred: {best_diag['mice_deterred']}")

    # Save JSON (include moves too)
    out = {
        "start_node": START_NODE,
        "best_path_nodes": best_diag["path_nodes"],
        "best_path_moves": best_moves[:best_diag.get("used_length", len(best_moves))].tolist(),
        "used_length": best_diag.get("used_length", len(best_moves)),
        "mice_paths": best_diag["mice_history"],
        "diagnostics": {
            k: v for k, v in best_diag.items() if k not in ("path_nodes", "mice_history")
        }
    }
    with open(BEST_JSON_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Saved] {BEST_JSON_PATH}")

    # Final plots
    plot_fitness(best_history)
    plot_path(best_diag["path_nodes"], title="Best path (final)")

if __name__ == "__main__":
    main()