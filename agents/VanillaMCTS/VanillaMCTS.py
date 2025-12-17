"""
Optimized Vanilla UCT MCTS Agent for Hex
- No Neural Networks (Pure Random Rollouts)
- No Heuristics (Pure UCT)
- Optimization: Incremental Union-Find for O(1) win checks during simulation
- Optimization: Fast integer-based board representation
"""

import time
import random
import math
from copy import deepcopy
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# --- Configuration ---
BOARD_SIZE = 11
UCT_C = 1.414  # Standard exploration constant

# --- Fast Data Structures ---


class FastUnionFind:
    """
    Optimized Disjoint Set Union (DSU) for fast connectivity checks.
    Manages integer sets representing board positions.
    """

    __slots__ = ["parent", "rank"]

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i):
        # Path compression
        root = i
        while root != self.parent[root]:
            root = self.parent[root]

        curr = i
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by rank
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
            return True
        return False


class FastHexState:
    """
    Lightweight Hex State for rapid random simulations.
    Uses integers: 0=Empty, 1=Red, 2=Blue
    """

    __slots__ = [
        "board",
        "uf_red",
        "uf_blue",
        "empty_cells",
        "winner",
        "size",
        "red_edges",
        "blue_edges",
    ]

    def __init__(self, size):
        self.size = size
        self.board = [0] * (size * size)
        self.empty_cells = list(range(size * size))
        self.winner = 0  # 0=None, 1=Red, 2=Blue

        # Virtual nodes for edges
        # Red connects Top(size*size) to Bottom(size*size+1)
        # Blue connects Left(size*size) to Right(size*size+1)
        # We use separate UF structures to keep them independent and fast
        self.uf_red = FastUnionFind(size * size + 2)
        self.uf_blue = FastUnionFind(size * size + 2)

        self.red_edges = (size * size, size * size + 1)
        self.blue_edges = (size * size, size * size + 1)

    def clone(self):
        """Create a deep copy for branching"""
        new_state = FastHexState(self.size)
        new_state.board = self.board[:]
        new_state.empty_cells = self.empty_cells[:]
        new_state.winner = self.winner

        # Deep copy UF internals
        new_state.uf_red.parent = self.uf_red.parent[:]
        new_state.uf_red.rank = self.uf_red.rank[:]
        new_state.uf_blue.parent = self.uf_blue.parent[:]
        new_state.uf_blue.rank = self.uf_blue.rank[:]

        return new_state

    def play(self, cell_idx, player):
        """Place a stone and update connectivity"""
        self.board[cell_idx] = player

        # Remove from empty cells (swap-remove is O(1))
        # Note: In pure random simulation we usually shuffle the list once and pop,
        # but for tree search we need specific moves.
        try:
            self.empty_cells.remove(cell_idx)
        except ValueError:
            pass  # Already removed (shouldn't happen in valid logic)

        # Update connectivity
        row = cell_idx // self.size
        col = cell_idx % self.size
        neighbors = []

        # Standard Hex Neighbors
        if row > 0:
            neighbors.append(cell_idx - self.size)  # Top
        if row < self.size - 1:
            neighbors.append(cell_idx + self.size)  # Bottom
        if col > 0:
            neighbors.append(cell_idx - 1)  # Left
        if col < self.size - 1:
            neighbors.append(cell_idx + 1)  # Right
        if row > 0 and col < self.size - 1:
            neighbors.append(cell_idx - self.size + 1)  # Top-Right
        if row < self.size - 1 and col > 0:
            neighbors.append(cell_idx + self.size - 1)  # Bot-Left

        uf = self.uf_red if player == 1 else self.uf_blue

        # Union with neighbors of same color
        for n in neighbors:
            if self.board[n] == player:
                uf.union(cell_idx, n)

        # Union with edges
        if player == 1:  # Red (Top-Bottom)
            if row == 0:
                uf.union(cell_idx, self.red_edges[0])
            if row == self.size - 1:
                uf.union(cell_idx, self.red_edges[1])

            if uf.find(self.red_edges[0]) == uf.find(self.red_edges[1]):
                self.winner = 1
        else:  # Blue (Left-Right)
            if col == 0:
                uf.union(cell_idx, self.blue_edges[0])
            if col == self.size - 1:
                uf.union(cell_idx, self.blue_edges[1])

            if uf.find(self.blue_edges[0]) == uf.find(self.blue_edges[1]):
                self.winner = 2

    def random_rollout(self, turn_player):
        """
        Play random moves until terminal.
        Returns winner (1 or 2).
        """
        if self.winner != 0:
            return self.winner

        # Shuffle remaining moves
        random.shuffle(self.empty_cells)

        # Use a local pointer for speed
        moves = self.empty_cells
        current = turn_player

        for move in moves:
            self.board[move] = current

            # Inline connectivity check for speed (copy-paste logic to avoid func call overhead)
            row = move // self.size
            col = move % self.size

            uf = self.uf_red if current == 1 else self.uf_blue

            # Neighbors
            ns = []
            if row > 0:
                ns.append(move - self.size)
            if row < self.size - 1:
                ns.append(move + self.size)
            if col > 0:
                ns.append(move - 1)
            if col < self.size - 1:
                ns.append(move + 1)
            if row > 0 and col < self.size - 1:
                ns.append(move - self.size + 1)
            if row < self.size - 1 and col > 0:
                ns.append(move + self.size - 1)

            for n in ns:
                if self.board[n] == current:
                    uf.union(move, n)

            # Edge check
            if current == 1:  # Red
                if row == 0:
                    uf.union(move, self.red_edges[0])
                if row == self.size - 1:
                    uf.union(move, self.red_edges[1])
                if uf.find(self.red_edges[0]) == uf.find(self.red_edges[1]):
                    return 1
            else:  # Blue
                if col == 0:
                    uf.union(move, self.blue_edges[0])
                if col == self.size - 1:
                    uf.union(move, self.blue_edges[1])
                if uf.find(self.blue_edges[0]) == uf.find(self.blue_edges[1]):
                    return 2

            current = 3 - current  # Swap 1 <-> 2

        return 0  # Should not happen in Hex


# --- MCTS Logic ---


class Node:
    __slots__ = [
        "parent",
        "move",
        "player",
        "children",
        "wins",
        "visits",
        "untried_moves",
    ]

    def __init__(self, parent, move, player, untried_moves):
        self.parent = parent
        self.move = move
        self.player = player  # The player who made 'move'
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = untried_moves  # Integer indices

    def uct_select_child(self):
        """Select child with highest UCT value"""
        # UCT = win_rate + C * sqrt(ln(parent_visits) / child_visits)
        best_score = -float("inf")
        best_child = None
        log_parent = math.log(self.visits)

        for child in self.children:
            # UCT from perspective of the parent (the player choosing the move)
            # So if child represents Opponent's move, we want to minimize opponent win?
            # Standard way: node.wins is wins for the player who Just Moved (node.player)
            # We want to choose the move that leads to US winning.

            # Pure UCT usually views stats from the perspective of the node's player.
            win_rate = child.wins / child.visits

            # Invert score if the child represents an opponent response?
            # Simplified: MCTS usually maximizes winrate for the player at THIS node.
            # If I am Red, I want to pick a child (Red Move) that has high Red Win Rate.

            ucb = win_rate + UCT_C * math.sqrt(log_parent / child.visits)

            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child

    def update(self, winner):
        self.visits += 1
        # If the winner matches the player who made the move to get to this node
        if winner == self.player:
            self.wins += 1
        elif winner == 0:  # Draw (rare)
            self.wins += 0.5


class VanillaMCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.my_int = 1 if colour == Colour.RED else 2
        self.time_limit = 290.0  # Total budget
        self.time_used = 0.0

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        start_time = time.time()

        # 1. Initialize Root State (Fast Version)
        root_state = self._board_to_fast_state(board)

        # Handle Swap Rule Logic for Root
        # (If we are turn 2 and swap is valid, the fast state handles it as just empty spots,
        # but we need to verify if we want to support swap in tree search)
        # For simplicity in Vanilla MCTS, we treat Swap as a move (-1, -1) if valid.

        untried = root_state.empty_cells[:]

        # Check if swap is available (Turn 2 for us, meaning board has exactly 1 stone)
        # In this specific Agent implementation, we check board directly
        stones_on_board = 0
        for i in range(root_state.size * root_state.size):
            if root_state.board[i] != 0:
                stones_on_board += 1

        can_swap = stones_on_board == 1 and self.colour == Colour.BLUE
        if can_swap:
            # We add a virtual index -1 to represent swap
            # But FastHexState needs to handle it.
            # For simplicity, if we swap, we just play on the transposed coordinate
            # of the opponent's stone.
            pass

        # Root represents the state BEFORE we pick a move
        # So root.player is the Opponent (who just moved)
        root_node = Node(
            parent=None, move=None, player=(3 - self.my_int), untried_moves=untried
        )

        # 2. Time Management
        remaining = self.time_limit - self.time_used
        # Simple heuristic: heavily weighted towards mid-game
        if stones_on_board < 10:
            alloc = remaining * 0.02
        elif stones_on_board < 40:
            alloc = remaining * 0.05
        else:
            alloc = remaining * 0.03

        alloc = max(0.5, min(alloc, 5.0))  # Caps
        end_time = start_time + alloc

        simulations = 0

        # 3. MCTS Loop
        while time.time() < end_time:
            node = root_node
            state = root_state.clone()

            # SELECTION (Traverse tree)
            while not node.untried_moves and node.children:
                node = node.uct_select_child()
                state.play(node.move, node.player)

            # EXPANSION (Add one node)
            if node.untried_moves:
                m = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
                current_player = 3 - node.player  # Swap turns
                state.play(m, current_player)

                new_node = Node(
                    parent=node,
                    move=m,
                    player=current_player,
                    untried_moves=state.empty_cells[:],
                )
                node.children.append(new_node)
                node = new_node

            # SIMULATION (Random Rollout)
            # Determine who plays next in the simulation
            next_sim_player = 3 - node.player
            winner = state.random_rollout(next_sim_player)

            # BACKPROPAGATION
            while node is not None:
                node.update(winner)
                node = node.parent

            simulations += 1

        # 4. Select Best Move
        # Robust child selection: most visits (not highest winrate)
        if not root_node.children:
            # Should not happen unless board full
            return Move(0, 0)  # Fallback

        best_child = max(root_node.children, key=lambda c: c.visits)

        elapsed = time.time() - start_time
        self.time_used += elapsed

        # Debug info
        # print(f"VanillaMCTS: {simulations} sims in {elapsed:.2f}s ({(simulations/elapsed):.0f} n/s)")

        # Convert integer index back to Move object
        idx = best_child.move
        r = idx // BOARD_SIZE
        c = idx % BOARD_SIZE
        return Move(r, c)

    def _board_to_fast_state(self, board: Board):
        """Convert object board to integer FastHexState"""
        state = FastHexState(board.size)
        for r in range(board.size):
            for c in range(board.size):
                clr = board.tiles[r][c].colour
                idx = r * board.size + c
                if clr == Colour.RED:
                    state.play(idx, 1)
                elif clr == Colour.BLUE:
                    state.play(idx, 2)
        return state
