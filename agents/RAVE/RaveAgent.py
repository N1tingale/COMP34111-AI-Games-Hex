"""
RAVE (Rapid Action Value Estimation) MCTS Agent for Hex.
- Uses AMAF (All-Moves-As-First) heuristic to learn faster.
- Blends UCT (exploitation) with RAVE values (exploration/heuristic).
- Optimized with incremental Union-Find and fast integer states.
"""

import time
import random
import math
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# --- Configuration ---
BOARD_SIZE = 11
RAVE_EQUIV = 3000  # RAVE equivalence parameter (k). Controls how long we trust RAVE.
                   # Higher = trust RAVE longer. Lower = switch to UCT sooner.

# --- Fast Data Structures (Reused from Vanilla for speed) ---

class FastUnionFind:
    __slots__ = ['parent', 'rank']
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i):
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
    """State optimized for RAVE: Tracks move history during rollout"""
    __slots__ = ['board', 'uf_red', 'uf_blue', 'empty_cells', 'winner', 'size', 'red_edges', 'blue_edges']

    def __init__(self, size):
        self.size = size
        self.board = [0] * (size * size)
        self.empty_cells = list(range(size * size))
        self.winner = 0 
        self.uf_red = FastUnionFind(size * size + 2)
        self.uf_blue = FastUnionFind(size * size + 2)
        self.red_edges = (size * size, size * size + 1)
        self.blue_edges = (size * size, size * size + 1)

    def clone(self):
        new_state = FastHexState(self.size)
        new_state.board = self.board[:]
        new_state.empty_cells = self.empty_cells[:]
        new_state.winner = self.winner
        new_state.uf_red.parent = self.uf_red.parent[:]
        new_state.uf_red.rank = self.uf_red.rank[:]
        new_state.uf_blue.parent = self.uf_blue.parent[:]
        new_state.uf_blue.rank = self.uf_blue.rank[:]
        return new_state

    def play(self, cell_idx, player):
        self.board[cell_idx] = player
        # Note: We assume cell_idx is removed from empty_cells by the caller 
        # or doesn't matter for the logic calling this specific play().
        
        # Incremental connectivity update
        row = cell_idx // self.size
        col = cell_idx % self.size
        neighbors = []
        if row > 0: neighbors.append(cell_idx - self.size)
        if row < self.size - 1: neighbors.append(cell_idx + self.size)
        if col > 0: neighbors.append(cell_idx - 1)
        if col < self.size - 1: neighbors.append(cell_idx + 1)
        if row > 0 and col < self.size - 1: neighbors.append(cell_idx - self.size + 1)
        if row < self.size - 1 and col > 0: neighbors.append(cell_idx + self.size - 1)

        uf = self.uf_red if player == 1 else self.uf_blue
        
        for n in neighbors:
            if self.board[n] == player:
                uf.union(cell_idx, n)
        
        if player == 1: 
            if row == 0: uf.union(cell_idx, self.red_edges[0])
            if row == self.size - 1: uf.union(cell_idx, self.red_edges[1])
            if uf.find(self.red_edges[0]) == uf.find(self.red_edges[1]):
                self.winner = 1
        else:
            if col == 0: uf.union(cell_idx, self.blue_edges[0])
            if col == self.size - 1: uf.union(cell_idx, self.blue_edges[1])
            if uf.find(self.blue_edges[0]) == uf.find(self.blue_edges[1]):
                self.winner = 2

    def random_rollout_rave(self, turn_player):
        """
        Rollout that records moves for AMAF updates.
        Returns: winner, p1_moves (set), p2_moves (set)
        """
        if self.winner != 0:
            return self.winner, set(), set()
            
        random.shuffle(self.empty_cells)
        moves = self.empty_cells
        current = turn_player
        
        # Track moves for RAVE
        p1_moves = set()
        p2_moves = set()

        for move in moves:
            self.board[move] = current
            
            if current == 1: p1_moves.add(move)
            else: p2_moves.add(move)

            # --- Inline Connectivity Check ---
            row = move // self.size
            col = move % self.size
            uf = self.uf_red if current == 1 else self.uf_blue
            
            ns = []
            if row > 0: ns.append(move - self.size)
            if row < self.size - 1: ns.append(move + self.size)
            if col > 0: ns.append(move - 1)
            if col < self.size - 1: ns.append(move + 1)
            if row > 0 and col < self.size - 1: ns.append(move - self.size + 1)
            if row < self.size - 1 and col > 0: ns.append(move + self.size - 1)

            for n in ns:
                if self.board[n] == current:
                    uf.union(move, n)
            
            if current == 1: 
                if row == 0: uf.union(move, self.red_edges[0])
                if row == self.size - 1: uf.union(move, self.red_edges[1])
                if uf.find(self.red_edges[0]) == uf.find(self.red_edges[1]):
                    return 1, p1_moves, p2_moves
            else:
                if col == 0: uf.union(move, self.blue_edges[0])
                if col == self.size - 1: uf.union(move, self.blue_edges[1])
                if uf.find(self.blue_edges[0]) == uf.find(self.blue_edges[1]):
                    return 2, p1_moves, p2_moves
            
            current = 3 - current
            
        return 0, p1_moves, p2_moves

# --- RAVE MCTS Node ---

class RAVENode:
    __slots__ = ['parent', 'move', 'player', 'children', 
                 'wins', 'visits', 
                 'amaf_wins', 'amaf_visits', 
                 'untried_moves']
    
    def __init__(self, parent, move, player, untried_moves):
        self.parent = parent
        self.move = move
        self.player = player
        self.children = []
        
        # Standard UCT Stats
        self.wins = 0.0
        self.visits = 0
        
        # RAVE (AMAF) Stats
        self.amaf_wins = 0.0
        self.amaf_visits = 0
        
        self.untried_moves = untried_moves

    def rave_select_child(self):
        """Select child using RAVE blended score"""
        best_score = -float('inf')
        best_child = None
        
        # Pre-calculate log for UCT
        # Avoid log(0)
        log_visits = math.log(self.visits) if self.visits > 0 else 0
        
        for child in self.children:
            # 1. UCT Score (Exploitation)
            if child.visits > 0:
                win_rate = child.wins / child.visits
                # Standard UCT Exploration
                exploration = 0.4 * math.sqrt(log_visits / child.visits)
                uct_val = win_rate + exploration
            else:
                # Prioritize unvisited nodes slightly less than winning nodes
                # but high enough to explore
                uct_val = 1.0 
            
            # 2. AMAF Score (Heuristic)
            if child.amaf_visits > 0:
                amaf_val = child.amaf_wins / child.amaf_visits
            else:
                amaf_val = 0.5
            
            # 3. Beta Calculation (The Blend)
            # beta approaches 0 as visits increase (trust UCT more)
            # beta approaches 1 when visits are low (trust RAVE more)
            # Formula: beta = sqrt(k / (3 * visits + k))
            if child.visits == 0:
                beta = 1.0
            else:
                beta = math.sqrt(RAVE_EQUIV / (3 * child.visits + RAVE_EQUIV))
            
            # 4. Final Blended Score
            score = (1.0 - beta) * uct_val + beta * amaf_val
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def update(self, winner, p1_moves, p2_moves):
        """Update both UCT and AMAF stats"""
        self.visits += 1
        
        # 1. Update UCT (Standard)
        if winner == self.player:
            self.wins += 1
        
        # 2. Update AMAF (RAVE) for Children
        # If a child's move was played by the winner LATER in the game,
        # we credit that child with an AMAF win.
        # This applies 'All-Moves-As-First' logic.
        
        winner_moves = p1_moves if winner == 1 else p2_moves
        
        for child in self.children:
            # We only update AMAF if the child represents a move played by the winner
            # AND the child represents the winner's colour.
            # child.player is the player who made the move to get to 'child'.
            if child.player == winner:
                if child.move in winner_moves:
                    child.amaf_visits += 1
                    child.amaf_wins += 1
                else:
                    # Move was NOT played by winner (or played by loser)
                    # We usually increment visits but not wins for the move 
                    # if it was played by the loser?
                    # Standard RAVE: Update stats for moves played by *either* side
                    # to track frequencies, but usually we focus on the "move's value".
                    
                    # Simplified AMAF: 
                    # If move was played by ANYONE, increment visits.
                    # If played by WINNER, increment wins.
                    move_in_p1 = child.move in p1_moves
                    move_in_p2 = child.move in p2_moves
                    
                    if move_in_p1 or move_in_p2:
                        child.amaf_visits += 1
                        # Win already handled above (only if child.player == winner)
                        # If child.player != winner, it's a loser's move, so 0 wins added.

class RAVEMCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.my_int = 1 if colour == Colour.RED else 2
        self.time_limit = 290.0
        self.time_used = 0.0

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        start_time = time.time()
        
        root_state = self._board_to_fast_state(board)
        
        # Handle Swap check (Turn 2 Blue)
        stones = 0
        for i in range(len(root_state.board)):
            if root_state.board[i] != 0: stones += 1
        if stones == 1 and self.colour == Colour.BLUE:
            # Naive swap handling: RAVE treats board as is. 
            # If we really want to swap, we rely on standard tree search finding it.
            pass

        # Root: Parent=None, Player=Opponent (so children are My Moves)
        root_node = RAVENode(parent=None, move=None, player=(3 - self.my_int), untried_moves=root_state.empty_cells[:])
        
        remaining = self.time_limit - self.time_used
        if stones < 10: alloc = remaining * 0.03
        elif stones < 50: alloc = remaining * 0.05
        else: alloc = remaining * 0.04
        alloc = max(0.2, min(alloc, 4.0))
        end_time = start_time + alloc

        simulations = 0
        
        while time.time() < end_time:
            node = root_node
            state = root_state.clone()
            
            # SELECTION
            while not node.untried_moves and node.children:
                node = node.rave_select_child()
                state.play(node.move, node.player)

            # EXPANSION
            if node.untried_moves:
                m = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
                current_player = 3 - node.player
                state.play(m, current_player)
                
                new_node = RAVENode(parent=node, move=m, player=current_player, untried_moves=state.empty_cells[:])
                node.children.append(new_node)
                node = new_node
            
            # SIMULATION (Returns moves for AMAF)
            next_player = 3 - node.player
            winner, p1_moves, p2_moves = state.random_rollout_rave(next_player)
            
            # BACKPROPAGATION (With RAVE Updates)
            while node is not None:
                node.update(winner, p1_moves, p2_moves)
                node = node.parent
            
            simulations += 1

        elapsed = time.time() - start_time
        self.time_used += elapsed
        
        if not root_node.children:
            return Move(0, 0)

        # Select best move based on Visits (Standard MCTS robustness)
        # RAVE is only used to GUIDE search, not for final selection.
        best_child = max(root_node.children, key=lambda c: c.visits)
        
        idx = best_child.move
        return Move(idx // BOARD_SIZE, idx % BOARD_SIZE)

    def _board_to_fast_state(self, board: Board):
        state = FastHexState(board.size)
        for r in range(board.size):
            for c in range(board.size):
                clr = board.tiles[r][c].colour
                idx = r * board.size + c
                if clr == Colour.RED:
                    state.play(idx, 1)
                    if idx in state.empty_cells: state.empty_cells.remove(idx)
                elif clr == Colour.BLUE:
                    state.play(idx, 2)
                    if idx in state.empty_cells: state.empty_cells.remove(idx)
        return state