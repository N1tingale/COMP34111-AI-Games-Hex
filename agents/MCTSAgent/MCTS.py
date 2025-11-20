from collections import deque
from enum import Enum, auto
from typing import TypedDict
from math import sqrt, log
import random
import copy
import sys
import os

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile

SQRT_TWO = sqrt(2)


class Node:
    def __init__(
        self,
        board: Board,
        current_colour: Colour,
        move: Move | None = None,
        parent: 'Node' = None,
    ):
        self.move = move
        self.parent = parent
        self.board = copy.deepcopy(board)
        self.current_colour = current_colour  # The color about to move
        self.visits = 0
        self.reward = 0.0
        self.children: dict[tuple[int, int], Node] = {}

        # Apply the move if provided
        if self.move and not self.move.is_swap():
            self.board.set_tile_colour(self.move.x, self.move.y, Colour.opposite(current_colour))
    
    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get all legal moves (empty tiles) for the current board state."""
        legal_moves = []
        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.tiles[i][j].colour is None:
                    legal_moves.append((i, j))
        return legal_moves
    
    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been expanded as children."""
        legal_moves = self.get_legal_moves()
        return len(self.children) >= len(legal_moves)
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        return self.board.has_ended(Colour.RED) or self.board.has_ended(Colour.BLUE)


class MCTS:
    def __init__(self, iterations: int = 2000) -> None:
        self.root: Node | None = None
        self.iterations = iterations

    def uct_value(self, node: Node, parent: Node, exploration_weight: float = SQRT_TWO) -> float:
        """Calculate the UCT (Upper Confidence Bound for Trees) value."""
        if node.visits == 0:
            return float('inf')
        
        exploitation = node.reward / node.visits
        exploration = exploration_weight * sqrt(log(parent.visits) / node.visits)
        return exploitation + exploration

    def select(self, node: Node) -> Node:
        """Select a node to expand using UCT until we reach a leaf node."""
        current = node
        
        while not current.is_terminal() and current.is_fully_expanded():
            # Select child with highest UCT value
            current = max(
                current.children.values(),
                key=lambda child: self.uct_value(child, current)
            )
        
        return current

    def expand(self, node: Node) -> Node:
        """Expand a node by adding a new child for an untried move."""
        if node.is_terminal():
            return node
        
        # Find untried moves
        legal_moves = node.get_legal_moves()
        tried_moves = set(node.children.keys())
        untried_moves = [move for move in legal_moves if move not in tried_moves]
        
        if not untried_moves:
            return node
        
        # Choose a random untried move
        move_tuple = random.choice(untried_moves)
        move = Move(move_tuple[0], move_tuple[1])
        
        # Create new child node
        child = Node(
            board=node.board,
            current_colour=Colour.opposite(node.current_colour),
            move=move,
            parent=node,
        )
        
        node.children[move_tuple] = child
        return child

    def simulate(self, node: Node) -> Colour:
        """Simulate a game from this node to completion using semi-random playouts."""
        # Copy the current state
        current_board = copy.deepcopy(node.board)
        current_colour = node.current_colour
        
        # Limit simulation depth to avoid excessive computation
        legal_moves = node.get_legal_moves()
        max_moves = min(len(legal_moves), 50)
        moves_made = 0
        
        # Play semi-random moves until game ends or move limit reached
        while moves_made < max_moves:
            # Check if game is over
            if current_board.has_ended(Colour.RED):
                return Colour.RED
            if current_board.has_ended(Colour.BLUE):
                return Colour.BLUE
            
            # Get legal moves
            legal_move_tuples = []
            for i in range(current_board.size):
                for j in range(current_board.size):
                    if current_board.tiles[i][j].colour is None:
                        legal_move_tuples.append((i, j))
            
            if not legal_move_tuples:
                break
            
            # Use heuristic-guided move selection (80% heuristic, 20% random)
            if random.random() < 0.8:
                move = self._heuristic_move(current_board, legal_move_tuples, current_colour)
            else:
                move = random.choice(legal_move_tuples)
            
            # Apply move
            current_board.set_tile_colour(move[0], move[1], current_colour)
            current_colour = Colour.opposite(current_colour)
            moves_made += 1
        
        # If simulation incomplete, check for winner
        if current_board.has_ended(Colour.RED):
            return Colour.RED
        if current_board.has_ended(Colour.BLUE):
            return Colour.BLUE
        
        # Use heuristic evaluation if no winner yet
        return self._evaluate_position(current_board, node.current_colour)
    
    def _heuristic_move(self, board: Board, legal_moves: list[tuple[int, int]], colour: Colour) -> tuple[int, int]:
        """Select a move using simple heuristics."""
        size = board.size
        best_move = None
        best_score = -float('inf')
        
        # Sample a subset of moves to evaluate (for performance)
        moves_to_check = random.sample(legal_moves, min(20, len(legal_moves)))
        
        for move in moves_to_check:
            score = 0
            x, y = move
            
            if colour == Colour.RED:
                # Red wants to connect top to bottom
                # Prefer moves in the center columns and progressing downward
                score += (size - abs(y - size // 2)) * 2  # Center preference
                score += x * 3  # Progress toward bottom
                
                # Bonus for connecting to existing red pieces
                for idx in range(Tile.NEIGHBOUR_COUNT):
                    nx = x + Tile.I_DISPLACEMENTS[idx]
                    ny = y + Tile.J_DISPLACEMENTS[idx]
                    if 0 <= nx < size and 0 <= ny < size:
                        if board.tiles[nx][ny].colour == Colour.RED:
                            score += 5
            else:
                # Blue wants to connect left to right
                # Prefer moves in the center rows and progressing rightward
                score += (size - abs(x - size // 2)) * 2  # Center preference
                score += y * 3  # Progress toward right
                
                # Bonus for connecting to existing blue pieces
                for idx in range(Tile.NEIGHBOUR_COUNT):
                    nx = x + Tile.I_DISPLACEMENTS[idx]
                    ny = y + Tile.J_DISPLACEMENTS[idx]
                    if 0 <= nx < size and 0 <= ny < size:
                        if board.tiles[nx][ny].colour == Colour.BLUE:
                            score += 5
            
            # Add randomness to avoid deterministic play
            score += random.random() * 2
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else random.choice(legal_moves)
    
    def _evaluate_position(self, board: Board, current_colour: Colour) -> Colour:
        """Evaluate a non-terminal position and return a pseudo-winner."""
        # Simple heuristic: count connectivity towards goal
        size = board.size
        
        red_score = 0
        blue_score = 0
        
        for i in range(size):
            for j in range(size):
                if board.tiles[i][j].colour == Colour.RED:
                    red_score += (i + 1)  # Reward progress toward bottom
                elif board.tiles[i][j].colour == Colour.BLUE:
                    blue_score += (j + 1)  # Reward progress toward right
        
        # Return winner based on better position
        if red_score > blue_score:
            return Colour.RED
        elif blue_score > red_score:
            return Colour.BLUE
        else:
            # Tie-break randomly
            return random.choice([Colour.RED, Colour.BLUE])

    def backpropagate(self, node: Node, result: Colour):
        """Backpropagate the simulation result up the tree."""
        current = node
        
        while current is not None:
            current.visits += 1
            
            # Update reward from the perspective of the player who just moved
            # The node's current_colour indicates whose turn it is AT this node (before the move)
            # So we check if the previous player (opposite of current turn) won
            if result == Colour.RED:
                if current.current_colour == Colour.BLUE:  # This node is after Red's move
                    current.reward += 1
            elif result == Colour.BLUE:
                if current.current_colour == Colour.RED:  # This node is after Blue's move
                    current.reward += 1
            
            current = current.parent

    def search(self, root: Node) -> tuple[int, int]:
        """Run MCTS from the root node and return the best move."""
        self.root = root
        
        for _ in range(self.iterations):
            # Selection
            leaf = self.select(self.root)
            
            # Expansion
            if not leaf.is_terminal() and leaf.visits > 0:
                leaf = self.expand(leaf)
            
            # Simulation
            result = self.simulate(leaf)
            
            # Backpropagation
            self.backpropagate(leaf, result)
        
        # Return move with highest visit count (most robust)
        if not self.root.children:
            legal_moves = self.root.get_legal_moves()
            return random.choice(legal_moves) if legal_moves else (0, 0)
        
        best_move = max(
            self.root.children.items(),
            key=lambda item: item[1].visits
        )[0]
        
        return best_move


class MoveType(Enum):
    START = auto()
    SWAP = auto()
    CHANGE = auto()


class Command(TypedDict):
    command: MoveType
    board: list[str]
    move: tuple[int, int] | None
    turn: int


def parse_input(input: str) -> Command:
    input_parts = input.strip().split(";")

    command = None

    if input_parts[0] == "START":
        command = MoveType.START
    elif input_parts[0] == "SWAP":
        command = MoveType.SWAP
    else:
        command = MoveType.CHANGE

    move = None

    if input_parts[1]:
        x, y = map(int, input_parts[1].split(","))
        move = (x, y)

    board_state_string = input_parts[2]
    board_rows = board_state_string.split(",")

    turn = int(input_parts[3])

    return Command(command=command, board=board_rows, move=move, turn=turn)


def board_from_strings(board_rows: list[str]) -> Board:
    """Convert list of string rows to Board object."""
    size = len(board_rows)
    board = Board(board_size=size)
    
    for i, row_str in enumerate(board_rows):
        for j, char in enumerate(row_str):
            board.tiles[i][j].colour = Colour.from_char(char)
    
    return board


def bfs(state: list[list[str]]) -> int:
    """
    DEPRECATED: Use Board.has_ended() instead.
    Check if there's a winning path using BFS.
    Returns: 0 if no winner, 1 if Red wins, 2 if Blue wins.
    """
    board = Board(board_size=len(state))
    for i, row in enumerate(state):
        for j, char in enumerate(row):
            board.tiles[i][j].colour = Colour.from_char(char)
    
    if board.has_ended(Colour.RED):
        return 1
    elif board.has_ended(Colour.BLUE):
        return 2
    return 0


def is_outcome(node: Node) -> int:
    """
    DEPRECATED: Use node.is_terminal() instead.
    Check if the current board state at a node is a finished game state.
    """
    if node.board.has_ended(Colour.RED):
        return 1
    elif node.board.has_ended(Colour.BLUE):
        return 2
    return 0


if __name__ == "__main__":
    mcts = MCTS(iterations=2000)
    
    while True:
        try:
            line = input()
            command_data = parse_input(line)
            
            # Convert board strings to Board object
            board = board_from_strings(command_data["board"])
            
            # Determine whose turn it is (Red goes first, turn 1 = Red)
            current_colour = Colour.RED if (command_data["turn"] % 2) == 1 else Colour.BLUE
            
            # Create root node
            root = Node(
                board=board,
                current_colour=current_colour,
                move=None,
            )
            
            # Run MCTS
            best_move = mcts.search(root)
            
            # Output the move
            print(f"{best_move[0]},{best_move[1]}")
            
        except EOFError:
            break
        except Exception as e:
            # In case of error, make a random valid move
            print("0,0")
