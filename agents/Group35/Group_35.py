"""
Tournament agent for Hex game using neural network + MCTS with advanced optimizations

ENHANCEMENTS ADDED:
1. BETTER TIME MANAGEMENT - Adaptive time allocation based on game phase and recent performance
2. IMMEDIATE WIN/MUST-BLOCK - Fast tactical checks using Union-Find connectivity analysis
3. CANDIDATE MOVE PRUNING - Focuses search on high-policy moves, limits branching factor
4. LOCALITY FILTERS - Prioritizes moves near recent play for better tactical awareness
5. ENHANCED UCB - Improved move selection with tie-breaking noise

Ive vomited out a "AlphaZero-style-ish" agent where:
- Neural network provides policy priors and value estimates
- MCTS explores smartly using these priors  
- Engineering optimizations make it efficient and strong
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
    """Convert board to 3-channel tensor (matching trainA100.py encoding)"""
    tensor = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            tile_colour = board.tiles[i][j].colour
            if tile_colour == current_player:
                tensor[0, i, j] = 1.0  # Current player's pieces
            elif tile_colour is not None:
                tensor[1, i, j] = 1.0  # Opponent's pieces
            else:
                tensor[2, i, j] = 1.0  # Empty positions
                
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
    """Monte Carlo Tree Search implementation with advanced optimizations"""
    
    def __init__(self, model, cpuct=1.4):
        self.model = model
        self.cpuct = cpuct
        self.transposition_table = {}  # Cache for position evaluations
        self.root = None  # For root reuse between moves
        self.last_board_hash = None
        
    def search(self, root_board, player_colour, time_limit=4.5, last_move=None):
        """Run MCTS search and return best move with advanced optimizations"""
        
        # 1. IMMEDIATE WIN/MUST-BLOCK DETECTION
        winning_moves = check_immediate_win(root_board, player_colour)
        if winning_moves:
            return winning_moves[0]  # Take any winning move immediately
        
        # Check if opponent has winning moves (must block)
        opp_winning_moves = check_immediate_win(root_board, Colour.opposite(player_colour))
        if len(opp_winning_moves) == 1:
            return opp_winning_moves[0]  # Block the only winning move
        
        # 2. ROOT REUSE - DISABLED for debugging
        # board_hash = board_to_hash(root_board)
        # root = self._get_reusable_root(board_hash, last_move)
        
        # Always create fresh root to avoid state corruption
        root = MCTSNode()
        
        # Expand root if needed
        if not root.is_expanded:
            self._expand_node(root, root_board, player_colour, last_move)
        
        # 3. ADAPTIVE TIME MANAGEMENT - More aggressive termination
        start_time = time.time()
        simulations = 0
        
        # Adaptive minimum simulations based on time budget
        if time_limit < 0.2:
            min_simulations = 20  # Very fast moves
        elif time_limit < 0.5:
            min_simulations = 50  # Fast moves
        elif time_limit < 1.0:
            min_simulations = 80  # Normal moves
        else:
            min_simulations = 120  # Slower moves
        
        # Maximum simulations to prevent runaway search
        max_simulations = min(2000, int(time_limit * 500))  # Scale with time
        
        while True:
            elapsed = time.time() - start_time
            
            # Hard time limit - always stop
            if elapsed >= time_limit:
                break
            
            # Stop if minimum simulations done and we're at 80% of time
            if simulations >= min_simulations and elapsed >= time_limit * 0.8:
                break
            
            # Stop if we hit max simulations
            if simulations >= max_simulations:
                break
            
            # Run one simulation
            node = root
            search_board = self._clone_board(root_board)
            current_player = player_colour
            
            # Selection: traverse down the tree
            while (not node.is_leaf() and 
                   not search_board.has_ended(Colour.RED) and 
                   not search_board.has_ended(Colour.BLUE)):
                action, node = self._select_child(node)
                self._apply_move(search_board, action, current_player)
                current_player = Colour.opposite(current_player)

            # Evaluation & Expansion
            if (not search_board.has_ended(Colour.RED) and 
                not search_board.has_ended(Colour.BLUE)):
                value = self._expand_node(node, search_board, current_player, last_move)
            else:
                # Terminal state
                value = 1.0 if search_board.get_winner() == player_colour else -1.0

            # Backpropagation
            self._backpropagate(node, value)
            simulations += 1
        
        # Store root for next search (disabled for debugging)
        # self.root = root
        # self.last_board_hash = board_hash
        
        return self._get_best_move(root)

    def _select_child(self, node):
        """Select child using enhanced UCB formula"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        # Add small noise to break ties and encourage exploration
        noise_factor = 0.001

        for action, child in node.children.items():
            q_value = -child.value()  # Negate for opponent perspective
            
            # Standard UCB exploration term
            u_score = (self.cpuct * child.prior * 
                      math.sqrt(node.visit_count) / (1 + child.visit_count))
            
            # Add small random noise to break ties
            noise = np.random.random() * noise_factor
            
            score = q_value + u_score + noise

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _expand_node(self, node, board, player_colour, last_move=None):
        """Expand node and return network evaluation with optimizations"""
        
        # TRANSPOSITION TABLE DISABLED for debugging
        # Always evaluate fresh to avoid cached state corruption
        policy, value = self._evaluate(board, player_colour)
        
        # Get candidate moves with various filters
        valid_moves = self._get_valid_actions(board)
        
        # CANDIDATE MOVE PRUNING - Use policy to focus on promising moves
        move_probs = []
        for move in valid_moves:
            idx = move_to_index(move)
            prob = float(policy[idx])
            move_probs.append((move, prob))
        
        # Sort by policy probability
        move_probs.sort(key=lambda x: x[1], reverse=True)
        
        # More aggressive move pruning based on game phase
        total_stones = sum(1 for i in range(board.size) for j in range(board.size) 
                          if board.tiles[i][j].colour is not None)
        
        if total_stones < 10:  # Early game - very aggressive pruning
            max_moves = 8
        elif total_stones < 30:  # Mid game - moderate pruning
            max_moves = 12
        else:  # Late game - less pruning for tactics
            max_moves = 18
        
        # Apply locality filter if we have a recent move
        if last_move is not None and not last_move.is_swap() and total_stones > 2:
            locality_moves = get_locality_filtered_moves(board, last_move, max_moves=max_moves)
            if locality_moves is not None:
                # Combine policy ranking with locality
                locality_set = set(locality_moves)
                filtered_moves = []
                
                # First add high-probability moves that are also local
                for move, prob in move_probs:
                    if move in locality_set and prob > 0.02:  # Slightly higher threshold
                        filtered_moves.append((move, prob))
                
                # Then add remaining high-probability moves (fewer in early game)
                prob_threshold = 0.08 if total_stones < 20 else 0.05
                for move, prob in move_probs:
                    if move not in locality_set and prob > prob_threshold:
                        filtered_moves.append((move, prob))
                
                # Ensure we have at least some moves but respect max_moves
                if len(filtered_moves) < max_moves // 2:
                    filtered_moves = move_probs[:max_moves]
                elif len(filtered_moves) > max_moves:
                    filtered_moves = filtered_moves[:max_moves]
                
                move_probs = filtered_moves
        else:
            # No locality filter, just use top policy moves with game-phase limits
            move_probs = move_probs[:max_moves]
        
        # Normalize probabilities
        action_priors = []
        total_prob = sum(prob for _, prob in move_probs)
        
        if total_prob > 0:
            for move, prob in move_probs:
                action_priors.append((move, prob / total_prob))
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(move_probs) if move_probs else 1.0
            action_priors = [(move, uniform_prob) for move, _ in move_probs]
        
        node.expand(action_priors)
        return value

    def _backpropagate(self, node, value):
        """Backup value through the tree"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent
            node = node.parent

    def _evaluate(self, board, current_player):
        """Evaluate position using neural network"""
        tensor_in = board_to_tensor(board, current_player)
        
        with torch.no_grad():
            policy_logits, value = self.model(tensor_in)
            
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).numpy()
        return policy_probs, value.item()

    def _get_reusable_root(self, board_hash, last_move):
        """Try to reuse previous search tree by finding matching subtree"""
        if self.root is None or last_move is None:
            return None
        
        # Look for a child node that matches the current position
        # This happens when the opponent played a move we considered
        for move, child in self.root.children.items():
            if move == last_move:
                # Found the subtree for this position
                # Make this child the new root
                child.parent = None
                return child
        
        return None
    
    def _get_best_move(self, root):
        """Select move with most visits, with additional heuristics"""
        if not root.children:
            return None
        
        # Get move with most visits
        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        
        # Additional safety check: if the best move has very few visits compared to second best,
        # and the value difference is small, prefer the move with higher policy prior
        children_by_visits = sorted(root.children.items(), key=lambda x: x[1].visit_count, reverse=True)
        
        if len(children_by_visits) >= 2:
            best_child = children_by_visits[0][1]
            second_child = children_by_visits[1][1]
            
            # If visit counts are close and values are close, prefer higher prior
            if (best_child.visit_count < second_child.visit_count * 2 and
                abs(best_child.value() - second_child.value()) < 0.1):
                if second_child.prior > best_child.prior * 1.5:
                    return children_by_visits[1][0]
        
        return best_move

    def _clone_board(self, board):
        """Create a copy of the board"""
        new_board = Board(board.size)
        for r in range(board.size):
            for c in range(board.size):
                new_board.tiles[r][c].colour = board.tiles[r][c].colour
        return new_board

    def _apply_move(self, board, move, colour):
        """Apply move to board (swap handled by game engine)"""
        # CRITICAL ISSUE: MCTS doesn't simulate swap at all!
        # Swap does nothing to board state, but player alternates, creating inconsistent search tree
        # TODO: Either handle swap properly in MCTS or remove swap from non-root decisions
        if not move.is_swap():
            board.set_tile_colour(move.x, move.y, colour)
    
    def _get_valid_actions(self, board):
        """Get all legal moves for current board state"""
        # CRITICAL ISSUE: Swap legality is too loose!
        # Currently allows swap whenever 1 stone on board, but should only be:
        # - Turn 2 AND second player (BLUE) in standard Hex rules
        # - This can cause illegal moves if framework checks turn number
        # TODO: Pass turn/colour info and gate: allow_swap = (turn == 2 and colour == BLUE)
        valid_moves = []
        occupied_count = 0
        
        for i in range(board.size):
            for j in range(board.size):
                if board.tiles[i][j].colour is None:
                    valid_moves.append(Move(i, j))
                else:
                    occupied_count += 1
                    
        # Swap available on turn 2 (exactly 1 stone on board)
        if occupied_count == 1:
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
