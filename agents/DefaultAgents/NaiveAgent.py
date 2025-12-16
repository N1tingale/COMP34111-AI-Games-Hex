from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class NaiveAgent(AgentBase):
    """This class describes the default Hex agent. It will randomly send a
    valid move at each turn, and it will choose to swap with a 50% chance.

    The class inherits from AgentBase, which is an abstract class.
    The AgentBase contains the colour property which you can use to get the agent's colour.
    You must implement the make_move method to make the agent functional.
    You CANNOT modify the AgentBase class, otherwise your agent might not function.
    """

    _choices: list[Move]
    _board_size: int = 11

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

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._choices = [
            (i, j) for i in range(self._board_size) for j in range(self._board_size)
        ]

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """The game engine will call this method to request a move from the agent.
        If the agent is to make the first move, opp_move will be None.
        If the opponent has made a move, opp_move will contain the opponent's move.
        If the opponent has made a swap move, opp_move will contain a Move object with x=-1 and y=-1,
        the game engine will also change your colour to the opponent colour.

        Args:
            turn (int): The current turn
            board (Board): The current board state
            opp_move (Move | None): The opponent's last move

        Returns:
            Move: The agent's move
        """

        # if turn == 2 and choice([0, 1]) == 1:
        if turn == 2:
            return Move(-1, -1)
        ##else:
            #x, y = choice(self._choices)
            #return Move(x, y)
            ##return choice(self._get_valid_actions(board))
        return choice(self._get_valid_actions(board))
