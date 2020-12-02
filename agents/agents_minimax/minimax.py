import math
from agents import common
from agents.common import check_end_state as game_state
from agents.common import BoardPiece, SavedState, PlayerAction
import numpy as np
from dataclasses import dataclass
from agents.common import GameState
from typing import Optional, List
from enum import Enum
from typing import Optional, Callable, Tuple

# chetno = greshno pri player2 - player (51 red)
# raboti za depth 2 pri player - player2 (51 red)
DEPTH = 4
ROW = 6
COL = 7
PLAYER_ON_TURN = BoardPiece
SavedValue = np.zeros(7)


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    value = alpha_beta_action(board, player)
    action = np.where(SavedValue == value)[0][0]
    print(action)
    return action,saved_state


def alpha_beta_action(board: np.ndarray, player: BoardPiece):
    """
    Main call function. Returns where player has to play
    """
    global PLAYER_ON_TURN
    PLAYER_ON_TURN = player
    return alpha_beta_minimax(board, player, -math.inf, math.inf, DEPTH)


def alpha_beta_minimax(board: np.ndarray, player: BoardPiece, alpha: int, beta: int, depth: int):
    """
    Alpha-beta pruning/Negamax. Called by alpha_beta_action
    """
    player2 = 0
    if player == BoardPiece(1):
        player2 = BoardPiece(2)
    else:
        player2 = BoardPiece(1)

    if (depth == 0 or
            game_state(board, BoardPiece(1)) != GameState.STILL_PLAYING or
            game_state(board, BoardPiece(2)) != GameState.STILL_PLAYING):
        if DEPTH == 1:
            return (depth + 1) * evaluate_curr_board(board, player)
        else : return (depth + 1) * evaluate_curr_board(board, player) + (depth + 1) * evaluate_curr_board(board, player2)

    if common.on_turn() == player:
        global SavedValue
        tmp_board = board.copy()
        for col in range(COL):
            if np.count_nonzero(board[:, col] == 0) > 0:
                after_action = common.apply_player_action(board, col, player, True)
                value = alpha_beta_minimax(after_action, player, alpha, beta, depth - 1)
                board = common.undo_move()
                if depth == DEPTH:
                    SavedValue[col] = value
                    board = tmp_board.copy()
                    print(SavedValue)
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return alpha

    else:
        for col in range(COL):
            if np.count_nonzero(board[:, col] == 0) > 0:
                # player
                after_action = common.apply_player_action(board, col, player2, True)
                # player 2
                value = alpha_beta_minimax(after_action, player, alpha, beta, depth - 1)
                board = common.undo_move()
                if value < beta:
                    beta = value
                    if alpha >= beta:
                        break
        return beta


def evaluate_curr_board(board: np.ndarray, player: BoardPiece) -> int:
    """

    """
    value_row_list = []
    for row in range(ROW):
        value_row_list.append(evaluate_position(board[row, :], player))
    value_row = np.sum(value_row_list)

    value_col_list = []
    for col in range(COL):
        value_col_list.append(evaluate_position(board[:, col].T, player))
    value_col = np.sum(value_col_list)

    value_main_diag_list = []
    for diag in range(-2, 4):
        main_diag = np.diag(board, diag)
        value_main_diag_list.append(evaluate_position(main_diag, player))
    value_main_diag = np.sum(value_main_diag_list)

    value_opp_diag_list = []
    for diag in range(-2, 4):
        opp_diag = np.diag(board[::-1], diag)
        value_opp_diag_list.append(evaluate_position(opp_diag, player))
    value_opp_diag = np.sum(value_opp_diag_list)

    return value_col + value_row + value_main_diag + value_opp_diag


def evaluate_position(array_from_board: np.ndarray, player: BoardPiece) -> int:
    """
    Returns evaluation of current row for the player
    """

    index = len(array_from_board) - 3


    sum_val = 0
    for i in range(index):
        tmp = array_from_board[i:i + 4]
        if np.count_nonzero(tmp == player) == 4:
            sum_val += 164
        if np.count_nonzero(tmp == player) == 3 and np.count_nonzero(tmp == BoardPiece(0)) == 1:
            sum_val += 40
        if np.count_nonzero(tmp == player) == 2 and np.count_nonzero(tmp == BoardPiece(0)) == 2:
            sum_val += 20
        if np.count_nonzero(tmp == player) == 1 and np.count_nonzero(tmp == BoardPiece(0)) == 3:
            sum_val += 1
    return sum_val
