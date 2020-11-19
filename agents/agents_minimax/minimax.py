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


def evaluate_curr_board(board: np.ndarray, player: BoardPiece) -> int:
    """

    """
    value_row_list = []
    for row in range(ROW):
        if np.count_nonzero(board[row, :] == 0) != 0:
            value_row_list.append(evaluate_position(board[row, :], player))
    value_row = np.max(value_row_list)

    value_col_list = []
    for col in range(COL):
        if np.count_nonzero(board[:, col] == 0) != 0:
            value_col_list.append(evaluate_position(board[:, col].T, player))
    value_col = np.max(value_col_list)

    value_main_diag_list = []
    for diag in range(-2, 5):
        main_diag = np.diag(board, diag)
        if np.count_nonzero(main_diag == 0) != 0:
            value_main_diag_list.append(evaluate_position(main_diag, player))
    value_main_diag = np.max(value_main_diag_list)

    value_opp_diag_list = []
    for diag in range(-2, 5):
        opp_diag = np.diag(board[::-1], diag)
        if np.count_nonzero(opp_diag == 0) != 0:
            value_opp_diag_list.append(evaluate_position(opp_diag, player))
    value_opp_diag = np.max(value_opp_diag_list)

    return max(value_col, value_row, value_main_diag, value_opp_diag)


def evaluate_position(array_from_board: np.ndarray, player: BoardPiece) -> int:
    """
    Returns evaluation of current row for the player
    """

    index = len(array_from_board) - 3

    list_of_values = []
    for i in range(index):
        tmp = array_from_board[i:i + 4]
        if np.count_nonzero(tmp == player) == 4:
            list_of_values.append(64)
        if np.count_nonzero(tmp == player) == 3 and np.count_nonzero(tmp == BoardPiece(0)) == 1:
            list_of_values.append(27)
        if np.count_nonzero(tmp == player) == 2 and np.count_nonzero(tmp == BoardPiece(0)) == 2:
            list_of_values.append(8)
        if np.count_nonzero(tmp == player) == 1 and np.count_nonzero(tmp == BoardPiece(0)) == 3:
            list_of_values.append(1)
    if len(list_of_values) == 0:
        return 0
    else:
        return np.max(list_of_values)



