import math
from agents import common
from agents.common import check_end_state as game_state
from agents.common import BoardPiece, SavedState, PlayerAction
import numpy as np
from dataclasses import dataclass
from agents.common import GameState
from typing import Optional, List
from enum import Enum

DEPTH = 4
ROW = 6
COL = 7


@dataclass
class Position:
    """
    board[x, y] with evaluation of the cell
    """

    # def __init__(self, x, y, value: Optional[int] = None):
    #     self.x = x
    #     self.y = y
    #     self.value = value

    x: int
    y: int
    value: Optional[int] = None

    def positions(self) -> (int, int, Optional[int]):
        return self.x, self.y, self.value


def alpha_beta_action(board: np.ndarray, player: BoardPiece) -> PlayerAction:
    """
    Main call function. Returns where player has to play
    """
    return alpha_beta_minimax(board, player, math.inf, -math.inf, DEPTH)


def alpha_beta_minimax(board: np.ndarray, player: BoardPiece, alpha: int, beta: int, depth: int) -> PlayerAction:
    """
    Alpha-beta pruning/Negamax. Called by alpha_beta_action
    """
    # if (depth == 0 or
    #         game_state(board, BoardPiece(1)) != GameState.STILL_PLAYING or
    #         game_state(board, BoardPiece(2)) != GameState.STILL_PLAYING):
    #     return
    pass


# X X X X -> 4**3 = 64
# X X X -> 3**3 = 27
# X X - X -> 3**3 = 27
# X X -> 2**3 = 8
# X - - X -> 2**3 = 8
# X - X -> 2**3 = 8
# X -> 1
def evaluate_curr_board(board: np.ndarray, player: BoardPiece) -> Position:
    """

    """
    value_row = 0
    for row in range(ROW):
        if np.count_nonzero(board[row, :] == 0) != 0 or np.count_nonzero(board[row, :] == 0) != COL:
            value_row = evaluate_row(board, board[row, :], player, row)

    value_col = 0
    for col in range(COL):
        if np.count_nonzero(board[:, col] == 0) != 0 or np.count_nonzero(board[:, col] == 0) != ROW:
            value_col = evaluate_column(board, board[:, col], player, col)


# X X X X -> 4**3 = 64 READY
# X X X -> 3**3 = 27
# X X - X -> 3**3 = 27 READY
# X X -> 2**3 = 8
# X - - X -> 2**3 = 8
# X - X -> 2**3 = 8
# X -> 1
def evaluate_row(board: np.ndarray, board_row: np.ndarray, player: BoardPiece, index_of_row: int) -> int:
    """
    Returns evaluation of current row for the player
    """
    list_of_pieces = where_are_my_pieces(board, player)
    i = 0
    while i < len(list_of_pieces):
        if list_of_pieces[i].x != index_of_row:
            list_of_pieces.remove(list_of_pieces[i])
            i -= 1
        else:
            i += 1

    if in_a_row(board_row, player) == 2 and len(list_of_pieces) > 2:
        if count_spaces(list_of_pieces, board_row, player, 1):
            return 27

    if in_a_row(board_row, player) == 4:
        return 64
#     DANooooooo
# galaaaaaaaaa

def evaluate_column(board_column: np.ndarray, player: BoardPiece, index_of_column: int) -> int:
    """
    Returns evaluation of current column for the player
    """
    pass


def evaluate_main_diagonal(board_main_diagonal: np.ndarray, player: BoardPiece) -> int:
    """
    Returns evaluation of current main diagonal for the player
    """
    pass


def evaluate_opposite_diagonal(board_opp_diagonal: np.ndarray, player: BoardPiece) -> int:
    """
    Returns evaluation of current opp. diagonal for the player
    """
    pass


def where_are_my_pieces(board: np.ndarray, player: BoardPiece) -> [Position]:
    list_of_pieces = []
    for row in range(ROW):
        for col in range(COL):
            if board[row, col] == player:
                pos = Position(x=row, y=col)
                list_of_pieces.append(pos)
    return list_of_pieces


def in_a_row(array: np.ndarray, player: BoardPiece) -> int:
    concatenate = np.concatenate(([False], array == player, [False])).astype(np.int8)
    is_player = np.diff(concatenate)
    return np.max(np.flatnonzero(is_player == -1) - np.flatnonzero(is_player == 1))


def count_spaces(list: [Position], board_row: np.ndarray, player: BoardPiece, spaces_needed: int) -> bool:
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            spaces = list[j].y - list[i].y - 1
            zero_count = np.count_nonzero(board_row[list[i].y:(list[j].y + 1)] == 0)
            if spaces == spaces_needed and zero_count == spaces:
                return True
    return False

