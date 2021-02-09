import math
from agents import common
from agents.common import check_end_state as game_state
from agents.common import BoardPiece, SavedState, PlayerAction
import numpy as np
from agents.common import GameState
from typing import Optional, Tuple

DEPTH = 4
ROW = 6
COL = 7


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    """
       Choose a move based on alpha-beta minimax
       Arguments:
       board: ndarray representation of the board
       player: whether agent plays with X (Player1) or O (Player2)
       saved_state: computation that it could reuse for future moves
       Return:
           Tuple[PlayerAction, SavedState]: returns the column, where the agent will play and agent computations
       """
    board_copy = board.copy()

    # compute with alpha-beta next action
    value, SavedValue = alpha_beta_minimax(board_copy, player, -math.inf, math.inf, DEPTH, True)
    action = np.where(SavedValue == value)[0][0]  # find the column comes the best value
    return action, saved_state


def alpha_beta_minimax(board: np.ndarray, player: BoardPiece, alpha: int, beta: int, depth: int, maximising: bool) \
        -> Tuple[int, np.ndarray]:
    """
    Recursive Alpha-beta pruning,minimax. Called by alpha_beta_action.
    Arguments:
            board: ndarray representation of the board
            player: the player or the opponent to be mini/maximized
            alpha: value set to -infinity
            beta: value set to infinity
            depth: evaluate the game tree down to some fixed depth
            maximising: True, when agent is on turn(MAX), False when opponent is on turn(MIN)
        Return:
            Tuple[int, SavedValue]: returns the best value and list with computed values for each column
    """

    # define the opponent
    opponent = other_player(player)

    # evaluate current board when all the moves are done or game is won by one of the players
    if (depth == 0 or
            game_state(board, BoardPiece(1)) != GameState.STILL_PLAYING or
            game_state(board, BoardPiece(2)) != GameState.STILL_PLAYING):
        return (depth + 1) * evaluate_curr_board(board, player) - (depth + 1) * evaluate_curr_board(board, opponent)


    SavedValue = np.zeros(7)  # list with computed values for each column
    # Alpha - Beta pruning
    # maximising the agent
    if maximising:
        tmp_board = board.copy()
        for col in range(COL):
            if np.count_nonzero(board[:, col] == 0) > 0:
                after_action = common.apply_player_action(board, col, player, True)  # do move with player
                value = alpha_beta_minimax(after_action, player, alpha, beta, depth - 1, False)  # evaluate board
                if depth == DEPTH:
                    SavedValue[col] = value  # saves values for child of root(first move)
                board = tmp_board.copy()
                # cut off
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
            else:
                SavedValue[col] = None  # when the column is full
        if depth == DEPTH:
            return alpha, SavedValue  # final return
        else:
            return alpha
    # minimising the opponent
    else:
        tmp_board = board.copy()
        for col in range(COL):
            if np.count_nonzero(board[:, col] == 0) > 0:
                after_action = common.apply_player_action(board, col, opponent, True)
                value = alpha_beta_minimax(after_action, player, alpha, beta, depth - 1, True)
                board = tmp_board.copy()
                # cut off
                if value < beta:
                    beta = value
                    if alpha >= beta:
                        break
        return beta


def evaluate_curr_board(board: np.ndarray, player: BoardPiece) -> int:
    """
           The function evaluates the board for the current player. Called by alpha_beta_minimax.
            Arguments:
                board: ndarray representation of the board
                player: the player whose moves have to be evaluated
            Return:
                int: sum of the evaluated values for each the column/row and diagonal
            """
    opponent = other_player(player)

    # when agent wins
    if common.check_end_state(board, player) == GameState.IS_WIN:
        return 110000
    # when opponent wins
    if common.check_end_state(board, opponent) == GameState.IS_WIN:
        return -100000

    # evaluate every row of the board
    value_row_list = []  # list with computed values for each row
    for row in range(ROW):
        value_row_list.append(evaluate_position(board[row, :], player))
    value_row = np.sum(value_row_list)  # sums the evaluated values for the all 6 rows

    # evaluate every column of the board
    value_col_list = []  # list with compute values for each column
    for col in range(COL):
        value_col_list.append(evaluate_position(board[:, col].T, player))
    value_col = np.sum(value_col_list)  # sums the evaluated values for the all 7 columns

    # evaluate every (left top, right bottom) diagonal of the board
    value_main_diag_list = []  # list with compute values for each (left top, right bottom) diagonal
    for diag in range(-2, 4):
        main_diag = np.diag(board, diag)
        value_main_diag_list.append(evaluate_position(main_diag, player))
        # sums the evaluated values for each (left top, right bottom)) diagonal
    value_main_diag = np.sum(value_main_diag_list)

    # evaluate every (right top, left bottom) diagonal of the board
    value_opp_diag_list = []  # list with compute values for each (right top, left bottom) diagonal
    for diag in range(-2, 4):
        opp_diag = np.diag(board[::-1], diag)
        value_opp_diag_list.append(evaluate_position(opp_diag, player))
    value_opp_diag = np.sum(value_opp_diag_list)  # sums the evaluated values for each (right top, left bottom) diagonal

    return value_col + value_row + value_main_diag + value_opp_diag


def evaluate_position(array_from_board: np.ndarray, player: BoardPiece) -> int:
    """
        This function calculates the heuristic value of the node.
        Called by evaluate_curr_board.
        Arguments:
            array_from_board: ndarray that represents specific column/diagonal/row of the board
            player: the player whose moves have to be evaluated
        Return:
            int: sum of the evaluated values for each subarray of array_from_board
        """
    # how many sequences of length 4 does the array contain
    index = len(array_from_board) - 3

    sum_val = 0  # value of the current array
    for i in range(index):
        tmp = array_from_board[i:i + 4]
        # when there are 3 pieces and space in between
        if np.count_nonzero(tmp == player) == 3 and np.count_nonzero(tmp == BoardPiece(0)) == 1:
            sum_val += 1000
        # when there are 2 pieces and two spaces in between
        if np.count_nonzero(tmp == player) == 2 and np.count_nonzero(tmp == BoardPiece(0)) == 2:
            sum_val += 100
        # when there is 1 piece and three spaces
        if np.count_nonzero(tmp == player) == 1 and np.count_nonzero(tmp == BoardPiece(0)) == 3:
            sum_val += 1
    return sum_val


def other_player(player: BoardPiece) -> BoardPiece:
    if player == BoardPiece(1):
        return BoardPiece(2)
    else:
        return BoardPiece(1)