import subprocess

from src.Colour import Colour
from src.AgentBase import AgentBase
from src.Move import Move
from src.Board import Board
from src.Game import logger

class MCTSAgent(AgentBase):
    def __init__(self, colour: Colour):
        super().__init__(colour)

        self.agent_process = subprocess.Popen(
            ["python3", "agents/MCTSAgent/MCTS.py"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def __getstate__(self):
        """
        Prepare the object for pickling (and deepcopy). Exclude the non-pickleable
        'agent_process' attribute.
        """
        state = self.__dict__.copy()
        if 'agent_process' in state:
            del state['agent_process']
        return state

    def __setstate__(self, state):
        """
        Restore the object after unpickling (and deepcopy). Re-create the 'agent_process'.
        """
        self.__dict__.update(state)
        self.agent_process = subprocess.Popen(
            ["python3", "agents/MCTSAgent/MCTS.py"],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

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

        rows = board.tiles
        board_strings = []
        for row in rows:
            row_string = ""
            for tile in row:
                colour = tile.colour
                if colour is None:
                    t = "0"
                else:
                    t = colour.get_char()
                row_string += t
            board_strings.append(row_string)
        board_string = ",".join(board_strings)

        if opp_move is None:
            command = f"START;;{board_string};{turn};"
        elif opp_move.is_swap():
            command = f"SWAP;;{board_string};{turn};"
        else:
            command = f"CHANGE;{opp_move.x},{opp_move.y};{board_string};{turn};"

        self.agent_process.stdin.write(command + "\n")
        self.agent_process.stdin.flush()

        response = self.agent_process.stdout.readline().rstrip()
        # assuming the response takes the form "x,y" with -1,-1 if the agent wants to make a swap move
        x, y = map(int, response.split(","))
        return Move(x, y)