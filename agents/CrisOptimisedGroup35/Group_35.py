"""
Tournament agent for Hex game using neural network + MCTS.

HARDWARE TARGET: Intel(R) Xeon(R) Gold 5416S (Sapphire Rapids)

OPTIMIZATIONS:
1. INT8 QUANTIZATION - Dynamic quantization for 2x inference speed.
2. JIT TRACE - Graph compilation for low overhead.
3. THREAD PINNING - Locked to 8 threads to match Docker limit & P-Cores.
4. MCTS TUNING - Heavy exploration (cpuct=2.0) for low-simulation regime.
5. OPTIMISED TIME MGMT - Aggressive time allocation based on game phase.
"""

import time
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# --- HARDWARE CONFIGURATION ---
torch.set_num_threads(8)
torch.backends.mkldnn.enabled = True

# --- GAME CONSTANTS ---
BOARD_SIZE = 11
NUM_BLOCKS = 10
NUM_FILTERS = 128
EMPTY = 0
RED_INT = 1
BLUE_INT = 2


class HexResNet(nn.Module):
    """Neural network architecture matching trainA100.py exactly"""

    def __init__(
        self, board_size=BOARD_SIZE, num_blocks=NUM_BLOCKS, num_filters=NUM_FILTERS
    ):
        super().__init__()
        self.board_size = board_size
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters),
                    nn.ReLU(),
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters),
                )
                for _ in range(num_blocks)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """Forward pass of the network."""
        out = self.conv_input(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.relu(out + residual)
        return self.policy_head(out), self.value_head(out)


# --- NUMPY HELPERS ---


def move_to_index(move):
    """Converts a Move object to a flat index."""
    if move.is_swap():
        return BOARD_SIZE * BOARD_SIZE
    return move.x * BOARD_SIZE + move.y


def board_to_numpy(board):
    """Converts the Board object to a numpy grid."""
    grid = np.zeros((board.size, board.size), dtype=np.int8)
    for r in range(board.size):
        for c in range(board.size):
            clr = board.tiles[r][c].colour
            if clr == Colour.RED:
                grid[r, c] = RED_INT
            elif clr == Colour.BLUE:
                grid[r, c] = BLUE_INT
    return grid


def numpy_to_tensor(grid, current_player_int):
    """Converts a numpy grid to a PyTorch tensor."""
    t = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    t[0] = grid == current_player_int
    t[1] = grid == (BLUE_INT if current_player_int == RED_INT else RED_INT)
    if current_player_int == RED_INT:
        t[2] = 1.0
    return torch.from_numpy(t).unsqueeze(0).to(memory_format=torch.channels_last)


# --- IMMEDIATE WIN CHECK ---
class UnionFind:
    """Simple Union-Find data structure."""

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        """Finds the representative of the set containing x."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Unions the sets containing x and y."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def check_immediate_win(board, colour):
    """
    Checks if there are any immediate winning moves for the given colour.
    Returns a list of winning moves.
    """

    size = board.size
    uf = UnionFind(size * size + 4)
    node_top, node_bot, node_left, node_right = (
        size * size,
        size * size + 1,
        size * size + 2,
        size * size + 3,
    )

    for i in range(size):
        for j in range(size):
            if board.tiles[i][j].colour == colour:
                pos = i * size + j
                if colour == Colour.RED:
                    if i == 0:
                        uf.union(pos, node_top)
                    if i == size - 1:
                        uf.union(pos, node_bot)
                else:
                    if j == 0:
                        uf.union(pos, node_left)
                    if j == size - 1:
                        uf.union(pos, node_right)
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < size
                        and 0 <= nj < size
                        and board.tiles[ni][nj].colour == colour
                    ):
                        uf.union(pos, ni * size + nj)

    winning_moves = []
    for i in range(size):
        for j in range(size):
            if board.tiles[i][j].colour is None:
                has_neighbor = False
                if colour == Colour.RED:
                    if i in (0, size - 1):
                        has_neighbor = True
                else:
                    if j in (0, size - 1):
                        has_neighbor = True

                if not has_neighbor:
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < size
                            and 0 <= nj < size
                            and board.tiles[ni][nj].colour == colour
                        ):
                            has_neighbor = True
                            break

                if has_neighbor:
                    pos = i * size + j
                    connected_edges = set()
                    if colour == Colour.RED:
                        if i == 0:
                            connected_edges.add(node_top)
                        if i == size - 1:
                            connected_edges.add(node_bot)
                    else:
                        if j == 0:
                            connected_edges.add(node_left)
                        if j == size - 1:
                            connected_edges.add(node_right)

                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < size
                            and 0 <= nj < size
                            and board.tiles[ni][nj].colour == colour
                        ):
                            connected_edges.add(uf.find(ni * size + nj))

                    target_start = node_top if colour == Colour.RED else node_left
                    target_end = node_bot if colour == Colour.RED else node_right

                    has_start = any(
                        uf.find(root) == uf.find(target_start)
                        for root in connected_edges
                    )
                    has_end = any(
                        uf.find(root) == uf.find(target_end) for root in connected_edges
                    )

                    if has_start and has_end:
                        winning_moves.append(Move(i, j))

    return winning_moves


def check_immediate_win_numpy(grid, player_int):
    """
    Checks if the given player has an immediate winning move on the numpy grid.
    Returns True if a winning move exists, False otherwise.
    """

    size = grid.shape[0]
    uf = UnionFind(size * size + 4)
    node_top, node_bot, node_left, node_right = (
        size * size,
        size * size + 1,
        size * size + 2,
        size * size + 3,
    )

    # 1. Build connectivity for existing stones
    for i in range(size):
        for j in range(size):
            if grid[i, j] == player_int:
                pos = i * size + j
                if player_int == RED_INT:
                    if i == 0:
                        uf.union(pos, node_top)
                    if i == size - 1:
                        uf.union(pos, node_bot)
                else:  # BLUE
                    if j == 0:
                        uf.union(pos, node_left)
                    if j == size - 1:
                        uf.union(pos, node_right)

                # Neighbors
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size and grid[ni, nj] == player_int:
                        uf.union(pos, ni * size + nj)

    # 2. Check if any empty spot connects the sides
    target_start = node_top if player_int == RED_INT else node_left
    target_end = node_bot if player_int == RED_INT else node_right

    start_root = uf.find(target_start)
    end_root = uf.find(target_end)

    for i in range(size):
        for j in range(size):
            if grid[i, j] == EMPTY:
                connected_start = False
                connected_end = False

                # Optimization: Check if it's on the edge itself
                if player_int == RED_INT:
                    if i == 0:
                        connected_start = True
                    if i == size - 1:
                        connected_end = True
                else:
                    if j == 0:
                        connected_start = True
                    if j == size - 1:
                        connected_end = True

                # Check neighbors if not already connected to edge
                if not (connected_start and connected_end):
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                        ni, nj = i + di, j + dj
                        if (
                            0 <= ni < size
                            and 0 <= nj < size
                            and grid[ni, nj] == player_int
                        ):
                            root = uf.find(ni * size + nj)
                            if root == start_root:
                                connected_start = True
                            if root == end_root:
                                connected_end = True
                            if connected_start and connected_end:
                                break

                if connected_start and connected_end:
                    return True

    return False


# --- MCTS CORE ---


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, parent=None, prior=0.0, move=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.move = move
        self.is_expanded = False

    def value(self):
        """Returns the mean value of the node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_priors):
        """Expands the node with the given action priors."""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob, action)
        self.is_expanded = True

    def is_leaf(self):
        """Returns True if the node is a leaf."""
        return not self.is_expanded


class MCTS:
    """Monte Carlo Tree Search implementation."""

    def __init__(self, model, cpuct=2.0):
        self.model = model
        self.cpuct = cpuct
        self.transposition_table = {}
        self.root = None

    def search(
        self, root_board, player_colour, time_limit, turn_number, last_move=None
    ):
        """
        Performs MCTS search.
        """

        winning_moves = check_immediate_win(root_board, player_colour)
        if winning_moves:
            return winning_moves[0]
        opp_winning_moves = check_immediate_win(
            root_board, Colour.opposite(player_colour)
        )
        if len(opp_winning_moves) == 1:
            return opp_winning_moves[0]

        if self.root is not None and last_move is not None:
            if last_move in self.root.children:
                self.root = self.root.children[last_move]
                self.root.parent = None
            else:
                self.root = None
        if self.root is None:
            self.root = MCTSNode()

        root_grid = board_to_numpy(root_board)
        player_int = RED_INT if player_colour == Colour.RED else BLUE_INT

        if not self.root.is_expanded:
            # Pass initial turn number to expand
            self._expand_node(self.root, root_grid, player_int, turn_number)

        start_time = time.time()
        simulations = 0
        min_sims = 2

        while True:
            if simulations > min_sims:
                if (time.time() - start_time) >= time_limit:
                    break

            node = self.root
            search_grid = root_grid.copy()
            current_player_int = player_int

            # Track turn depth in simulation to prevent illegal swaps deep in tree
            sim_turn = turn_number

            while not node.is_leaf():
                action, node = self._select_child(node)
                self._apply_move_numpy(search_grid, action, current_player_int)
                current_player_int = 3 - current_player_int
                sim_turn += 1

            # Check for immediate win at leaf
            if check_immediate_win_numpy(search_grid, current_player_int):
                value = 1.0
            else:
                value = self._expand_node(
                    node, search_grid, current_player_int, sim_turn
                )

            self._backpropagate(node, value)
            simulations += 1

        if not self.root.children:
            return None

        best_move = self._get_best_move(self.root)

        if best_move in self.root.children:
            self.root = self.root.children[best_move]
            self.root.parent = None
        else:
            self.root = None

        return best_move

    def _select_child(self, node):
        best_score = -float("inf")
        best_action = None
        best_child = None
        current_cpuct = self.cpuct
        sqrt_visit = math.sqrt(node.visit_count)

        for action, child in node.children.items():
            score = -child.value() + (
                current_cpuct * child.prior * sqrt_visit / (1 + child.visit_count)
            )
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand_node(self, node, grid, player_int, turn_number):
        board_hash = grid.tobytes() + bytes([player_int])

        if board_hash in self.transposition_table:
            policy, value = self.transposition_table[board_hash]
        else:
            tensor_in = numpy_to_tensor(grid, player_int)
            with torch.inference_mode():
                policy_logits, value_tensor = self.model(tensor_in)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
            value = value_tensor.item()
            self.transposition_table[board_hash] = (policy, value)

        valid_indices = np.argwhere(grid == EMPTY)
        valid_moves = [Move(r, c) for r, c in valid_indices]

        # Strict swap validity check
        if np.count_nonzero(grid) == 1 and turn_number == 2:
            valid_moves.append(Move(-1, -1))

        move_probs = []
        for move in valid_moves:
            idx = move_to_index(move)
            prob = float(policy[idx])
            move_probs.append((move, prob))
        move_probs.sort(key=lambda x: x[1], reverse=True)

        filtered_moves = []
        cumulative_prob = 0.0
        for move, prob in move_probs:
            filtered_moves.append((move, prob))
            cumulative_prob += prob
            if cumulative_prob > 0.95 and len(filtered_moves) >= 3:
                break

        total_p = sum(p for _, p in filtered_moves)
        if total_p > 0:
            action_priors = [(m, p / total_p) for m, p in filtered_moves]
        else:
            action_priors = [(m, 1.0 / len(filtered_moves)) for m, _ in filtered_moves]

        node.expand(action_priors)
        return value

    def _backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def _get_best_move(self, root):
        if not root.children:
            return None
        return max(
            root.children.keys(),
            key=lambda m: (root.children[m].visit_count, root.children[m].prior),
        )

    def _apply_move_numpy(self, grid, move, player_int):
        """
        Apply move to numpy grid.
        MATCHES mcts_hex.cpp:
        - If Swap: Remove old stone, place NEW stone at (c, r) as current player.
        """
        if move.is_swap():
            # 1. Find the opponent's stone (matches C++ logic)
            rows, cols = np.nonzero(grid)

            if len(rows) > 0:
                # Get current coordinates (r, c)
                r, c = rows[0], cols[0]

                # 2. Clear the old stone (matches C++ line 94: grid[p1_move] = EMPTY)
                grid[r, c] = 0

                # 3. Transpose and Place (matches C++ line 107: grid[new_move] = current_player)
                # We place the stone at (c, r)
                grid[c, r] = player_int
        else:
            # Standard move
            grid[move.x, move.y] = player_int


class TournamentAgent(AgentBase):
    """
    Tournament agent for Hex game using neural network + MCTS.
    """

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.model = HexResNet()
        self.time_budget = 290.0
        self.time_used = 0.0
        self.load_model()
        self.mcts = MCTS(self.model)

    def load_model(self):
        """Loads the best available model from disk."""
        agent_dir = Path(__file__).parent
        model_paths = [
            agent_dir / "model_hpc.pt",
            Path("model_hpc.pt"),
        ]

        loaded = False
        for path in model_paths:
            if path.exists():
                try:
                    state = torch.load(str(path), map_location="cpu")
                    new_state = {}
                    for k, v in state.items():
                        k = k.replace("module.", "").replace("_orig_mod.", "")
                        new_state[k] = v
                    self.model.load_state_dict(new_state, strict=True)
                    self.model.eval()
                    print(f"Loaded {path}")
                    loaded = True
                    break
                except Exception:
                    pass

        if not loaded:
            print("WARNING: Using random weights")

        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"Quantization failed: {e}")

        try:
            dummy = torch.randn(1, 3, BOARD_SIZE, BOARD_SIZE)
            self.model = torch.jit.trace(self.model, dummy)
        except Exception as e:
            print(f"JIT failed: {e}")

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Selects the best move using MCTS.
        """
        # 1. Game Start Reset
        if turn in (1, 2):
            self.time_used = 0.0
            self.mcts.root = None
            self.mcts.transposition_table = {}

        # 2. Clear tree on opponent swap
        # If the opponent swapped, the "Simulated World" (which used transposition)
        # no longer matches the "Real World" (which swapped roles).
        # We clear the tree to prevent "Ghost Stone" illegal moves.
        if opp_move is not None and opp_move.is_swap():
            self.mcts.root = None
            self.mcts.transposition_table = {}

        # 3. Time Management
        empty = sum(
            1 for r in range(11) for c in range(11) if board.tiles[r][c].colour is None
        )

        # 5.0s safety buffer
        rem_time = max(1.0, self.time_budget - self.time_used)
        est_moves = max(1, empty // 2)
        base = rem_time / est_moves

        if empty > 100:
            # OPENING (Moves 1-10): Fast.
            mult = 0.5
        elif empty > 80:
            # EARLY MID (Moves 10-20): Ramp up.
            mult = 1.0
        elif empty > 40:
            # THE CRUNCH (Moves 20-40): MAXIMUM POWER.
            mult = 1.6
        elif empty > 20:
            # LATE MID (Moves 40-50): Taper down.
            mult = 1.0
        else:
            # ENDGAME (Moves 50+): Sprint.
            mult = 0.6

        # Hard limits to ensure we never stall or play instantly
        limit = max(0.4, min(7.0, base * mult))

        # 4. Search
        start = time.time()

        # Pass the turn number to search so it knows when Swap is legal
        best = self.mcts.search(board, self.colour, limit, turn, opp_move)
        elapsed = time.time() - start

        self.time_used += elapsed
        # time_left = self.time_budget - self.time_used

        if self.mcts.root:
            # speed = self.mcts.root.visit_count / elapsed if elapsed > 0 else 0
            # print(
            #     f"Move {turn} | Time Left: {time_left:.1f}s | "
            #     f"Sims: {self.mcts.root.visit_count} | Speed: {speed:.1f} sims/s"
            # )
            pass

        # 5. Fallback
        if best is None:
            for r in range(11):
                for c in range(11):
                    if board.tiles[r][c].colour is None:
                        return Move(r, c)

        return best
