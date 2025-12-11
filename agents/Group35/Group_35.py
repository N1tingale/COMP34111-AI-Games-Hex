"""Tournament agent for Hex game using neural network + MCTS"""
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


class MCTSNode:
    """MCTS node for tree search"""
    
    def __init__(self, parent=None, prior=0.0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        
    def value(self):
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_priors):
        """Expand node with children for legal moves"""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = MCTSNode(self, prob)

class MCTS:
    """Monte Carlo Tree Search implementation"""
    
    def __init__(self, model, cpuct=1.4):
        self.model = model
        self.cpuct = cpuct
        
    def search(self, root_board, player_colour, time_limit=4.5):
        """Run MCTS search and return best move"""
        root = MCTSNode()
        self._expand_node(root, root_board, player_colour)
        
        start_time = time.time()
        while time.time() - start_time < time_limit:
            node = root
            search_board = self._clone_board(root_board)
            current_player = player_colour
            
            # Selection: traverse down the tree
            while (node.children and 
                   not search_board.has_ended(Colour.RED) and 
                   not search_board.has_ended(Colour.BLUE)):
                action, node = self._select_child(node)
                self._apply_move(search_board, action, current_player)
                current_player = Colour.opposite(current_player)

            # Evaluation & Expansion
            if (not search_board.has_ended(Colour.RED) and 
                not search_board.has_ended(Colour.BLUE)):
                value = self._expand_node(node, search_board, current_player)
            else:
                # Terminal state
                value = 1.0 if search_board.get_winner() == player_colour else -1.0

            # Backpropagation
            self._backpropagate(node, value)
            
        return self._get_best_move(root)

    def _select_child(self, node):
        """Select child using UCB formula"""
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            q_value = -child.value()  # Negate for opponent perspective
            u_score = (self.cpuct * child.prior * 
                      math.sqrt(node.visit_count) / (1 + child.visit_count))
            score = q_value + u_score

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _expand_node(self, node, board, player_colour):
        """Expand node and return network evaluation"""
        policy, value = self._evaluate(board, player_colour)
        valid_moves = self._get_valid_actions(board)
        
        action_priors = []
        total_prob = 0
        
        for move in valid_moves:
            idx = move_to_index(move)
            prob = float(policy[idx])
            action_priors.append((move, prob))
            total_prob += prob
            
        # Normalize probabilities over valid moves
        if total_prob > 0:
            action_priors = [(m, p / total_prob) for m, p in action_priors]
        
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

    def _get_best_move(self, root):
        """Select move with most visits"""
        if not root.children:
            return None
        return max(root.children.keys(), key=lambda m: root.children[m].visit_count)

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
    """Tournament agent using neural network + MCTS"""
    
    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.model = HexResNet()
        self.time_budget = 180.0
        self.time_used = 0.0
        self.load_model()
        self.mcts = MCTS(self.model)
        
    def load_model(self):
        """Load model with exact trainA100.py compatibility"""
        from pathlib import Path
        
        # Robust path handling - look relative to this file
        agent_dir = Path(__file__).parent
        model_paths = [
            agent_dir / "model_hpc.pt",
            agent_dir / "model.pt", 
            Path("model_hpc.pt"),  # Fallback to CWD
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
        """Make move using MCTS with neural network guidance"""
        # POTENTIAL ISSUE: Move coordinate convention risk
        # We use Move(i,j) and board.tiles[i][j] consistently, but if framework
        # interprets Move(x,y) differently from tiles[row][col], could cause illegal moves
        # TODO: Verify Move(x,y) matches board.tiles[x][y] convention
        
        # Reset time tracker at game start
        if turn in (1, 2):
            self.time_used = 0.0

        # Calculate time allocation
        empty_tiles = sum(1 for r in range(board.size) for c in range(board.size) 
                         if board.tiles[r][c].colour is None)
        
        est_moves_left = (empty_tiles / 2) + 4
        remaining_time = max(0.5, self.time_budget - self.time_used - 2.0)
        time_limit = max(0.1, min(4.5, remaining_time / est_moves_left))

        # MISSING: Turn info not passed to MCTS for proper swap legality
        # TODO: Pass allow_swap flag based on turn==2 and self.colour==BLUE
        # allow_swap = (turn == 2 and self.colour == Colour.BLUE)
        # best_move = self.mcts.search(board, self.colour, time_limit, allow_swap)
        
        # Run MCTS search
        start_time = time.time()
        best_move = self.mcts.search(board, self.colour, time_limit=time_limit)
        self.time_used += time.time() - start_time
        
        # Fallback to any legal move if MCTS fails
        if best_move is None:
            for i in range(board.size):
                for j in range(board.size):
                    if board.tiles[i][j].colour is None:
                        return Move(i, j)
                        
        return best_move
