import subprocess
import re
import os

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger


class MoHexAgent(AgentBase):
    """
    Agent that uses MoHex (Monte Carlo Hex) engine.
    MoHex is a strong Hex-playing program that uses Monte Carlo Tree Search.
    """
    
    def __init__(self, colour: Colour, mohex_path: str = None, time_limit: float = 5.0):
        super().__init__(colour)
        
        # Default paths to check for MoHex
        if mohex_path is None:
            possible_paths = [
                "mohex",  # If in PATH
                "./mohex",
                "./agents/MoHexAgent/mohex",
                "C:/Program Files/MoHex/mohex.exe",
                "/usr/local/bin/mohex",
            ]
            mohex_path = None
            for path in possible_paths:
                try:
                    # Test if the path works
                    test_proc = subprocess.Popen(
                        [path],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    test_proc.terminate()
                    mohex_path = path
                    break
                except (FileNotFoundError, OSError):
                    continue
            
            if mohex_path is None:
                raise FileNotFoundError(
                    "MoHex executable not found. Please install MoHex and either:\n"
                    "1. Add it to your PATH\n"
                    "2. Place it in agents/MoHexAgent/\n"
                    "3. Specify the path when creating MoHexAgent"
                )
        
        self.mohex_path = mohex_path
        self.time_limit = time_limit
        self.board_size = 11
        
        # Start MoHex process
        self.process = subprocess.Popen(
            [self.mohex_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Initialize MoHex
        self._send_command("boardsize 11")
        self._send_command("clear_board")
        
        # Set time limit
        self._send_command(f"time_settings 0 {time_limit} 1")
    
    def _send_command(self, command: str) -> str:
        """Send a command to MoHex and get the response."""
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            
            # Read response until we get "= " or "? "
            response_lines = []
            while True:
                line = self.process.stdout.readline()
                if not line:
                    break
                response_lines.append(line)
                # Check if this is the final response line
                if line.startswith("=") or line.startswith("?"):
                    # Read one more line (usually empty)
                    self.process.stdout.readline()
                    break
            
            response = "".join(response_lines)
            return response
        except Exception as e:
            logger.error(f"Error communicating with MoHex: {e}")
            return ""
    
    def _coord_to_gtp(self, x: int, y: int) -> str:
        """Convert board coordinates to GTP format (e.g., (0,0) -> 'a1')."""
        # GTP uses columns a-z and rows 1-n
        col = chr(ord('a') + y)
        row = str(x + 1)
        return col + row
    
    def _gtp_to_coord(self, gtp: str) -> tuple[int, int]:
        """Convert GTP format to board coordinates (e.g., 'a1' -> (0,0))."""
        gtp = gtp.lower().strip()
        col = ord(gtp[0]) - ord('a')
        row = int(gtp[1:]) - 1
        return (row, col)
    
    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """
        Make a move using MoHex.
        
        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move
            
        Returns:
            Move: The agent's move
        """
        
        # If opponent made a move, tell MoHex
        if opp_move is not None and not opp_move.is_swap():
            opp_colour = "black" if self.colour == Colour.BLUE else "white"
            move_gtp = self._coord_to_gtp(opp_move.x, opp_move.y)
            self._send_command(f"play {opp_colour} {move_gtp}")
        
        # Get MoHex's move
        our_colour = "black" if self.colour == Colour.RED else "white"
        response = self._send_command(f"genmove {our_colour}")
        
        # Parse the response
        # Response format: "= <move>\n\n"
        match = re.search(r'= ([a-z]\d+)', response.lower())
        if match:
            move_gtp = match.group(1)
            x, y = self._gtp_to_coord(move_gtp)
            return Move(x, y)
        else:
            # Fallback: make a random legal move
            logger.warning(f"Could not parse MoHex response: {response}")
            for i in range(board.size):
                for j in range(board.size):
                    if board.tiles[i][j].colour is None:
                        return Move(i, j)
            return Move(0, 0)
    
    def __del__(self):
        """Clean up MoHex process."""
        try:
            if hasattr(self, 'process') and self.process:
                self._send_command("quit")
                self.process.terminate()
                self.process.wait(timeout=1)
        except:
            pass
    
    def __getstate__(self):
        """Prepare the object for pickling (and deepcopy). Exclude the non-pickleable process."""
        state = self.__dict__.copy()
        if 'process' in state:
            del state['process']
        return state
    
    def __setstate__(self, state):
        """Restore the object after unpickling (and deepcopy). Re-create the process."""
        self.__dict__.update(state)
        
        # Restart MoHex process
        self.process = subprocess.Popen(
            [self.mohex_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Re-initialize
        self._send_command("boardsize 11")
        self._send_command("clear_board")
        self._send_command(f"time_settings 0 {self.time_limit} 1")
