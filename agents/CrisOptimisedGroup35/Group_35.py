"""
Tournament agent for Hex game using neural network + MCTS with advanced optimizations

ENHANCEMENTS ADDED (CRIS VERSION):
1. Removed Locality Pruning - "Ladder" escapes often require playing on the other side of the board
2. Cumulative Probability Pruning - Select moves until cumulative probability > 97%. 
   The agent looks at 2 moves when the choice is obvious, and 20+ moves when uncertain.
3. Enable Tree Reuse - Retain MCTS tree between moves to save computation and build on prior search.
4. Fast Board Hashing - Use numpy byte representation of stone planes for quick hashing in transposition table.
"""
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# Constants matching trainA100.py exactly
BOARD_SIZE = 11
NUM_BLOCKS = 10
NUM_FILTERS = 128


class HexResNet(nn.Module):
    """Neural network architecture matching trainA100.py exactly"""
    
    def __init__(self, board_size=BOARD_SIZE, num_blocks=NUM_BLOCKS, num_filters=NUM_FILTERS):
        super().__init__()
        self.board_size = board_size
        
        # Input convolution: 3 channels -> num_filters
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        
        # Residual blocks (exactly matching trainA100.py)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
            )
            for _ in range(num_blocks)
        ])
        
        # Policy head: outputs board_size^2 + 1 (121 + 1 = 122)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1),
        )
        
        # Value head: outputs single value (matching trainA100.py exactly)
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
        """Forward pass matching trainA100.py exactly"""
        out = self.conv_input(x)
        for block in self.res_blocks:
            residual = out
            out = block(out)
            out = F.relu(out + residual)
        return self.policy_head(out), self.value_head(out)

def move_to_index(move):
    """Convert move to policy index (matching trainA100.py)"""
    if move.is_swap():
        return BOARD_SIZE * BOARD_SIZE  # Index 121
    return move.x * BOARD_SIZE + move.y


def board_to_tensor(board, current_player):
    """
    Convert board to 3-channel tensor matching mcts_hex.cpp encoding.
    Plane 0: My stones
    Plane 1: Opponent stones
    Plane 2: Player Color (All 1.0 if Red/Vertical, All 0.0 if Blue/Horizontal)
    """
    tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            tile_colour = board.tiles[i][j].colour
            
            if tile_colour == current_player:
                tensor[0, i, j] = 1.0  # Plane 0: My stones
            elif tile_colour is not None:
                tensor[1, i, j] = 1.0  # Plane 1: Opponent stones
                
    # Fix: Plane 2 is Player Color (Orientation), not Empty spots
    if current_player == Colour.RED:
        tensor[2, :, :] = 1.0
    else:
        tensor[2, :, :] = 0.0
        
    return torch.FloatTensor(tensor).unsqueeze(0)


def board_to_hash(board):
    """Create hash key for transposition table"""
    hash_str = ""
    for i in range(board.size):
        for j in range(board.size):
            colour = board.tiles[i][j].colour
            if colour is None:
                hash_str += "0"
            elif colour == Colour.RED:
                hash_str += "R"
            else:
                hash_str += "B"
    return hash_str


class UnionFind:
    """Union-Find data structure for connectivity analysis"""
    
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def check_immediate_win(board, colour):
    """Check if colour has an immediate winning move using Union-Find"""
    size = board.size
    
    # Create virtual nodes for edges
    # For RED: top edge = size*size, bottom edge = size*size+1
    # For BLUE: left edge = size*size+2, right edge = size*size+3
    uf = UnionFind(size * size + 4)
    
    # Connect existing pieces
    for i in range(size):
        for j in range(size):
            if board.tiles[i][j].colour == colour:
                pos = i * size + j
                
                # Connect to virtual edges
                if colour == Colour.RED:
                    if i == 0:  # Top edge
                        uf.union(pos, size * size)
                    if i == size - 1:  # Bottom edge
                        uf.union(pos, size * size + 1)
                else:  # BLUE
                    if j == 0:  # Left edge
                        uf.union(pos, size * size + 2)
                    if j == size - 1:  # Right edge
                        uf.union(pos, size * size + 3)
                
                # Connect to adjacent pieces
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size:
                        if board.tiles[ni][nj].colour == colour:
                            uf.union(pos, ni * size + nj)
    
    # Check each empty position for immediate win
    winning_moves = []
    for i in range(size):
        for j in range(size):
            if board.tiles[i][j].colour is None:
                pos = i * size + j
                temp_uf = UnionFind(size * size + 4)
                
                # Copy existing connections
                for x in range(size * size + 4):
                    temp_uf.parent[x] = uf.parent[x]
                    temp_uf.rank[x] = uf.rank[x]
                
                # Add this move
                if colour == Colour.RED:
                    if i == 0:
                        temp_uf.union(pos, size * size)
                    if i == size - 1:
                        temp_uf.union(pos, size * size + 1)
                else:
                    if j == 0:
                        temp_uf.union(pos, size * size + 2)
                    if j == size - 1:
                        temp_uf.union(pos, size * size + 3)
                
                # Connect to adjacent pieces
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, -1), (-1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < size and 0 <= nj < size:
                        if board.tiles[ni][nj].colour == colour:
                            temp_uf.union(pos, ni * size + nj)
                
                # Check if this creates a winning path
                if colour == Colour.RED:
                    if temp_uf.find(size * size) == temp_uf.find(size * size + 1):
                        winning_moves.append(Move(i, j))
                else:
                    if temp_uf.find(size * size + 2) == temp_uf.find(size * size + 3):
                        winning_moves.append(Move(i, j))
    
    return winning_moves


def get_locality_filtered_moves(board, last_move, max_moves=20):
    """Get moves filtered by locality to recent play - optimized version"""
    if last_move is None or last_move.is_swap():
        return None
    
    size = board.size
    lx, ly = last_move.x, last_move.y
    
    # Use a more efficient approach - expand outward from last move
    local_moves = []
    
    # Start with immediate neighbors (distance 1)
    for radius in range(1, min(4, size)):  # Limit search radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:  # Skip inner radii already processed
                    continue
                    
                x, y = lx + dx, ly + dy
                if (0 <= x < size and 0 <= y < size and 
                    board.tiles[x][y].colour is None):
                    local_moves.append(Move(x, y))
                    
                    # Early termination if we have enough moves
                    if len(local_moves) >= max_moves:
                        return local_moves
    
    return local_moves if local_moves else None


class MCTSNode:
    """MCTS node for tree search with enhanced features"""
    
    def __init__(self, parent=None, prior=0.0, move=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.move = move  # The move that led to this node
        self.is_expanded = False
        
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_priors):
        """Expand node with children for legal moves"""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob, action)
        self.is_expanded = True
    
    def is_leaf(self):
        """Check if this is a leaf node"""
        return not self.is_expanded
    
    def get_path_moves(self):
        """Get sequence of moves from root to this node"""
        moves = []
        node = self
        while node.parent is not None:
            if node.move is not None:
                moves.append(node.move)
            node = node.parent
        return list(reversed(moves))

class MCTS:
    """Monte Carlo Tree Search - UNLOCKED VERSION (Stronger)"""
    
    def __init__(self, model, cpuct=1.4):
        self.model = model
        self.cpuct = cpuct
        self.transposition_table = {}  # Enabled
        self.root = None
        
    def search(self, root_board, player_colour, time_limit=4.5, last_move=None):
        """Run MCTS search and return best move"""
        
        # 1. IMMEDIATE WIN/MUST-BLOCK CHECK
        winning_moves = check_immediate_win(root_board, player_colour)
        if winning_moves:
            return winning_moves[0]
        
        opp_winning_moves = check_immediate_win(root_board, Colour.opposite(player_colour))
        if len(opp_winning_moves) == 1:
            return opp_winning_moves[0]
        
        # 2. TREE REUSE (Improvement #3)
        # If we have a root from last turn, try to move down the tree
        # to match the opponent's move.
        if self.root is not None and last_move is not None:
            if last_move in self.root.children:
                self.root = self.root.children[last_move]
                self.root.parent = None # Detach to save memory
            else:
                self.root = None # Opponent played unexpected move, reset
        
        # If no root (first move or reset), create new
        if self.root is None:
            self.root = MCTSNode()
            
        # Expand root if needed
        if not self.root.is_expanded:
            self._expand_node(self.root, root_board, player_colour)
        
        # 3. SEARCH LOOP
        start_time = time.time()
        simulations = 0
        
        # Dynamic simulation constraints based on time
        min_sims = 50 if time_limit < 0.5 else 100
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= time_limit:
                break
            # Optimization: Stop early if we have done enough sims and time is tight
            if simulations > min_sims and elapsed > time_limit * 0.85:
                break
            
            node = self.root
            search_board = self._clone_board(root_board)
            current_player = player_colour
            
            # SELECTION
            while (not node.is_leaf() and 
                   not search_board.has_ended(Colour.RED) and 
                   not search_board.has_ended(Colour.BLUE)):
                action, node = self._select_child(node)
                self._apply_move(search_board, action, current_player)
                current_player = Colour.opposite(current_player)

            # EVALUATION & EXPANSION
            if (not search_board.has_ended(Colour.RED) and 
                not search_board.has_ended(Colour.BLUE)):
                value = self._expand_node(node, search_board, current_player)
            else:
                # Terminal state
                # If winner is the player who just moved (previous player), value is +1 for them
                # But 'value' variable is usually from perspective of current_player
                winner = search_board.get_winner()
                value = 1.0 if winner == current_player else -1.0

            # BACKPROPAGATION
            self._backpropagate(node, value)
            simulations += 1
        
        # 4. SELECT BEST MOVE
        best_move = self._get_best_move(self.root)
        
        # PREPARE ROOT FOR NEXT TURN (Tree Reuse)
        # We move our root down to the move WE chose
        if best_move in self.root.children:
            self.root = self.root.children[best_move]
            self.root.parent = None
        else:
            self.root = None
            
        return best_move

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Slightly higher cpuct for more exploration in complex positions
        current_cpuct = self.cpuct 

        for action, child in node.children.items():
            q_value = -child.value() # Flip perspective
            
            u_score = (current_cpuct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count))
            
            score = q_value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _expand_node(self, node, board, player_colour):
        """Expansion with Transposition Table and Cumulative Pruning"""
        
        # 1. CREATE TENSOR
        tensor_in = board_to_tensor(board, player_colour)
        
        # 2. FAST HASHING (Improvement #4)
        # Hash the bytes of the stone planes (0 and 1) only. 
        # This is extremely fast compared to string building.
        board_hash = tensor_in[0, :2].cpu().numpy().tobytes()
        
        if board_hash in self.transposition_table:
            policy, value = self.transposition_table[board_hash]
        else:
            with torch.no_grad():
                policy_logits, value_tensor = self.model(tensor_in)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
            value = value_tensor.item()
            # Cache it
            self.transposition_table[board_hash] = (policy, value)
        
        # 3. GET VALID MOVES
        valid_moves = self._get_valid_actions(board)
        
        # 4. CUMULATIVE PROBABILITY PRUNING (Improvement #2)
        # Instead of "Top 8", we take the top moves that sum to 97% probability
        move_probs = []
        for move in valid_moves:
            idx = move_to_index(move)
            prob = float(policy[idx])
            move_probs.append((move, prob))
            
        # Sort by high probability
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        filtered_moves = []
        cumulative_prob = 0.0
        
        # Always take at least the top 2 moves, even if one is 99%
        min_moves = 2
        # Cutoff threshold
        prob_threshold = 0.97
        
        for move, prob in move_probs:
            filtered_moves.append((move, prob))
            cumulative_prob += prob
            if len(filtered_moves) >= min_moves and cumulative_prob > prob_threshold:
                break
        
        # Normalize the priors of the selected moves so they sum to 1
        total_filtered_prob = sum(p for _, p in filtered_moves)
        action_priors = []
        if total_filtered_prob > 0:
            for m, p in filtered_moves:
                action_priors.append((m, p / total_filtered_prob))
        else:
            # Fallback (rare)
            uniform = 1.0 / len(filtered_moves)
            action_priors = [(m, uniform) for m, _ in filtered_moves]
            
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
        # Simply return most visited (Standard AlphaZero)
        return max(root.children.keys(), key=lambda m: root.children[m].visit_count)

    def _clone_board(self, board):
        new_board = Board(board.size)
        for r in range(board.size):
            for c in range(board.size):
                new_board.tiles[r][c].colour = board.tiles[r][c].colour
        return new_board

    def _apply_move(self, board, move, colour):
        """Standard apply move with correct Swap/Transpose logic"""
        if move.is_swap():
            prev_move = None
            found_stone = False
            for r in range(board.size):
                for c in range(board.size):
                    if board.tiles[r][c].colour is not None:
                        prev_move = (r, c)
                        board.tiles[r][c].colour = None # Remove existing
                        found_stone = True
                        break
                if found_stone: break
            
            if found_stone:
                # Transpose and place as current player
                board.set_tile_colour(prev_move[1], prev_move[0], colour)
        else:
            board.set_tile_colour(move.x, move.y, colour)
    
    def _get_valid_actions(self, board):
        valid_moves = []
        occupied = False
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    valid_moves.append(Move(i, j))
                else:
                    occupied = True
        
        # Check if swap is valid (exactly 1 stone on board)
        # Count accurately
        count = 0
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is not None:
                    count += 1
        
        if count == 1:
            valid_moves.append(Move(-1, -1))
            
        return valid_moves

class TournamentAgent(AgentBase):
    """Tournament agent using neural network + MCTS with advanced optimizations"""
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.model = HexResNet()
        self.time_budget = 290.0  # More conservative total budget
        self.time_used = 0.0
        self.move_times = []  # Track time per move for better allocation
        self.last_move = None  # Track last move for root reuse and locality
        self.moves_made = 0  # Track number of moves made
        self.load_model()
        self.mcts = MCTS(self.model)
        
    def load_model(self):
        """Load model with exact trainA100.py compatibility"""
        from pathlib import Path
        
        # Robust path handling - look relative to this file
        agent_dir = Path(__file__).parent
        model_paths = [
            agent_dir / "model_hpc_latest.pt",  # Use latest model for fair comparison
            agent_dir / "model_hpc.pt",
            agent_dir / "model.pt", 
            Path("model_hpc_latest.pt"),  # Fallback to CWD
            Path("model_hpc.pt"),
            Path("model.pt")
        ]
        
        for model_path in model_paths:
            if not model_path.exists():
                continue
                
            try:
                # Load state dict with trainA100.py prefix handling
                state_dict = torch.load(str(model_path), map_location="cpu")
                new_state_dict = {}
                
                for k, v in state_dict.items():
                    # Handle DataParallel prefix
                    if k.startswith("module."):
                        k = k[7:]
                    # Handle torch.compile prefix  
                    if k.startswith("_orig_mod."):
                        k = k[10:]
                    new_state_dict[k] = v
                    
                self.model.load_state_dict(new_state_dict, strict=True)
                self.model.eval()
                print(f"Loaded model from {model_path}")
                return
                
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
                
        print("WARNING: No model found. Using random weights.")
        self.model.eval()

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make move using enhanced MCTS with advanced optimizations"""
        
        # Reset tracking at game start
        if turn in (1, 2):
            self.time_used = 0.0
            self.move_times = []
            self.last_move = None
            self.moves_made = 0
            # Clear MCTS state for new game
            self.mcts.root = None
            self.mcts.transposition_table = {}

        # IMPROVED TIME MANAGEMENT - Much more aggressive limits
        empty_tiles = sum(1 for r in range(board.size) for c in range(board.size) 
                         if board.tiles[r][c].colour is None)
        
        # Calculate remaining time with safety buffer
        safety_buffer = 10.0  # Keep 10 seconds as emergency reserve
        remaining_time = max(1.0, self.time_budget - self.time_used - safety_buffer)
        
        # Estimate moves we'll make (roughly half of remaining empty tiles)
        est_our_moves_left = max(1, empty_tiles // 2)
        
        # Base time per move
        base_time_per_move = remaining_time / est_our_moves_left
        
        # Game phase multipliers - much more conservative
        if empty_tiles > 90:  # Very early opening
            phase_multiplier = 0.3  # Very fast opening moves
        elif empty_tiles > 70:  # Opening
            phase_multiplier = 0.5  # Fast opening
        elif empty_tiles > 40:  # Midgame
            phase_multiplier = 0.8  # Moderate midgame
        elif empty_tiles > 15:  # Late midgame
            phase_multiplier = 1.2  # More time for tactics
        else:  # Endgame
            phase_multiplier = 1.5  # Most time for critical endgame
        
        # Adaptive adjustment based on recent performance
        if len(self.move_times) >= 3:
            recent_avg = sum(self.move_times[-3:]) / 3
            target_avg = base_time_per_move * phase_multiplier
            
            if recent_avg > target_avg * 1.5:
                # We're too slow, speed up significantly
                adaptive_multiplier = 0.6
            elif recent_avg > target_avg:
                # We're a bit slow, speed up
                adaptive_multiplier = 0.8
            elif recent_avg < target_avg * 0.3:
                # We're very fast, can afford more time
                adaptive_multiplier = 1.3
            else:
                # We're on track
                adaptive_multiplier = 1.0
        else:
            adaptive_multiplier = 1.0
        
        # Calculate final time limit with hard caps
        target_time = base_time_per_move * phase_multiplier * adaptive_multiplier
        
        # Hard limits to prevent runaway timing
        min_time = 0.05  # Minimum 50ms per move
        max_time = min(3.0, remaining_time * 0.3)  # Max 3s or 30% of remaining time
        
        time_limit = max(min_time, min(max_time, target_time))
        
        # Run enhanced MCTS search
        start_time = time.time()
        best_move = self.mcts.search(
            board, 
            self.colour, 
            time_limit=time_limit,
            last_move=opp_move
        )
        
        elapsed_time = time.time() - start_time
        self.time_used += elapsed_time
        self.move_times.append(elapsed_time)
        self.moves_made += 1
        
        # Keep only recent move times for adaptive learning
        if len(self.move_times) > 8:
            self.move_times = self.move_times[-8:]
        
        # Debug timing info (can be removed for tournament)
        if elapsed_time > 1.0:
            print(f"Move {self.moves_made}: {elapsed_time:.2f}s (target: {time_limit:.2f}s, remaining: {self.time_budget - self.time_used:.1f}s)")
        
        # Fallback to any legal move if MCTS fails
        if best_move is None:
            # Try to find a reasonable fallback move
            # 1. Look for moves near center
            center = board.size // 2
            for radius in range(3):
                for i in range(max(0, center - radius), min(board.size, center + radius + 1)):
                    for j in range(max(0, center - radius), min(board.size, center + radius + 1)):
                        if board.tiles[i][j].colour is None:
                            return Move(i, j)
            
            # 2. Any legal move
            for i in range(board.size):
                for j in range(board.size):
                    if board.tiles[i][j].colour is None:
                        return Move(i, j)
        
        # Update last move for next iteration
        self.last_move = best_move
        return best_move
