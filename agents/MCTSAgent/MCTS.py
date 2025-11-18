from collections import deque
from enum import Enum, auto
from typing import TypedDict
from math import sqrt, log

SQRT_TWO = sqrt(2)
STARTING_POSITION = []

for i in range(11):
    STARTING_POSITION.append(["0"] * 11)


STARTING_LEGAL_MOVES = []

for i in range(11):
    for j in range(11):
        STARTING_LEGAL_MOVES.append((i, j))


class Node:
    def __init__(
        self,
        board_position: list[list[str]],
        legal_moves: list[tuple[int, int]],
        turn_red: bool = True,
        move: tuple[int, int] | None = None,
    ):
        self.move = move
        self.board_position = board_position
        self.visited = False
        self.visits = 0
        self.reward = 0
        self.legal_moves: list[tuple[int, int]] = legal_moves
        self.children: dict[tuple[int, int], Node] = {}

        if self.move:
            self.legal_moves.remove(self.move)

            # TODO: Implement swap
            if move == (-1, -1):
                ...

            elif turn_red:
                self.board_position[self.move[0]][self.move[1]] = "R"
            else:
                self.board_position[self.move[0]][self.move[1]] = "B"


class MCTS:
    def __init__(self) -> None:
        self.root: Node = Node(
            board_position=STARTING_POSITION, legal_moves=STARTING_LEGAL_MOVES
        )

    def tree_policy(self, action: Node) -> float:
        return action.reward / action.visits + SQRT_TWO * sqrt(
            log(self.root.visits) / action.visits
        )

    def select(self) -> Node:
        current_node = self.root

        unvisited_child = None

        while not unvisited_child:
            for child in current_node.children.values():
                if not child.visited:
                    unvisited_child = child

            best_score_key = max(
                current_node.children,
                key=lambda x: self.tree_policy(current_node.children[x]),
            )

            current_node = current_node.children[best_score_key]

        return unvisited_child

    def expand(self, action: Node):
        if len(action.legal_moves) != len(action.children):
            for move in action.legal_moves:
                if not action.children.get(move, None):
                    action.children[move] = Node(
                        move=move,
                        legal_moves=action.legal_moves,
                        board_position=action.board_position,
                    )


class MoveType(Enum):
    START = auto()
    SWAP = auto()
    CHANGE = auto()


class Command(TypedDict):
    command: MoveType
    board: list[list[str]]
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
    board = []

    for row in board_state_string.split(","):
        board.append(row)

    turn = int(input_parts[3])

    return Command(command=command, board=board, move=move, turn=turn)


def bfs(state: list[list[str]]) -> int:
    # TODO: Implement BFS to check if either a red path from top to bottom exists, or a blue path from left to right
    ...


def is_outcome(node: Node):
    # TODO: Check if the current board state at a node is a finished game state
    ...


if __name__ == "__main__":
    input = input()
    res = parse_input(input)

    print(res["board"])
    print(res["command"])
    print(res["move"])
    print(res["turn"])
