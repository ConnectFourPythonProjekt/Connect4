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
PLAYER_ON_TURN = BoardPiece
SavedValue = np.zeros(7)  # list with compute values for every column


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
    value = alpha_beta_action(board, player)  # compute alpha-beta value
    action = np.where(SavedValue == value)[0][0]  # find in from which column comes the best value
    return action, saved_state


def alpha_beta_action(board: np.ndarray, player: BoardPiece):
    """
       Main call function calling minimax alpha beta pruning

        Arguments:
            board: ndarray representation of the board
            player: the agent to be maximized
        Return:
            int: the final value

        """
    # sets current player on turn
    global PLAYER_ON_TURN
    PLAYER_ON_TURN = player
    return alpha_beta_minimax(board, player, -math.inf, math.inf, DEPTH)


def alpha_beta_minimax(board: np.ndarray, player: BoardPiece, alpha: int, beta: int, depth: int):
    """
    Recursive Alpha-beta pruning,minimax. Called by alpha_beta_action.
    Arguments:
            board: ndarray representation of the board
            player: the player or the opponent to be mini/maximized
            alpha: value set to -infinity
            beta: value set to infinity
            depth: evaluate the game tree down to some fixed depth
        Return:
            int: the final value for the evaluation
    """

    # define the opponent
    player2 = 0
    if player == BoardPiece(1):
        player2 = BoardPiece(2)
    else:
        player2 = BoardPiece(1)

    # evaluate current board when all the moves are done or game is won by one of the players
    if (depth == 0 or
            game_state(board, BoardPiece(1)) != GameState.STILL_PLAYING or
            game_state(board, BoardPiece(2)) != GameState.STILL_PLAYING):
        return (depth + 1) * evaluate_curr_board(board, player) - (depth + 1) * evaluate_curr_board(board, player2)

    # checks for current player win
    global SavedValue
    board_before = board.copy()
    if block(board, player2) > -1:
        board = board_before.copy()
        if depth == DEPTH:
            SavedValue[block(board, player2)] = depth * 100000000  # if player can win on the first move
        return depth * 100000000

    # check for opponent win
    if block(board, player) > -1 and depth > 0:
        board = board_before.copy()
        if depth == DEPTH:
            # if you have to block the opponent on the first move
            SavedValue[block(board, player)] = -(depth * 100000000)
        return -(depth * 100000000)

    # Alpha - Beta pruning
    # maximizing
    if common.on_turn() == player:
        tmp_board = board.copy()
        for col in range(COL):
            if np.count_nonzero(board[:, col] == 0) > 0:
                after_action = common.apply_player_action(board, col, player, True)  # do move with player
                value = alpha_beta_minimax(after_action, player, alpha, beta, depth - 1)  # recursively evaluate board
                board = common.undo_move()
                if depth == DEPTH:
                    SavedValue[col] = value  # saves values for child of root(first move)
                    board = tmp_board.copy()
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return alpha
    # minimizing
    else:
        for col in range(COL):
            if np.count_nonzero(board[:, col] == 0) > 0:
                after_action = common.apply_player_action(board, col, player2, True)
                value = -alpha_beta_minimax(after_action, player, alpha, beta, depth - 1)
                board = common.undo_move()
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

    # evaluate every row of the board
    value_row_list = []  # list with compute values for each row
    for row in range(ROW):
        value_row_list.append(evaluate_position(board[row, :], player))
    value_row = np.sum(value_row_list)  # sums the values for the all 6 rows

    # evaluate every column of the board
    value_col_list = []  # list with compute values for each column
    for col in range(COL):
        value_col_list.append(evaluate_position(board[:, col].T, player))
    value_col = np.sum(value_col_list)  # sums the values for the all 7 columns

    # evaluate every (left top, right bottom) diagonal of the board
    value_main_diag_list = []  # list with compute values for each (left top, right bottom) diagonal
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
        This function calculates the heuristic value of the node.
        Called by evaluate_curr_board and block.

        Arguments:
            array_from_board: ndarray that represents specific column/diagonal/row of the board
            player: the player whose moves have to be evaluated
        Return:
            int: sum of the evaluated values for each subarray of array_from_board

        """
    # how many 4 sequences are in the array
    index = len(array_from_board) - 3

    sum_val = 0  # score of the current array
    for i in range(index):
        tmp = array_from_board[i:i + 4]
        # when there is a win
        if np.count_nonzero(tmp == player) == 4:
            sum_val += 100000
        # when there are 3 pieces with space between
        if np.count_nonzero(tmp == player) == 3 and np.count_nonzero(tmp == BoardPiece(0)) == 1:
            sum_val += 1000
        # when there are 2 pieces with space between
        if np.count_nonzero(tmp == player) == 2 and np.count_nonzero(tmp == BoardPiece(0)) == 2:
            sum_val += 100
        # when there is 1 piece with space between
        if np.count_nonzero(tmp == player) == 1 and np.count_nonzero(tmp == BoardPiece(0)) == 3:
            sum_val += 1
    return sum_val


def block(board: np.ndarray, player: BoardPiece) -> int:
    """
           The function checks if the opponent of player can win in only one move.
           Called by alpha_beta_minimax

           Arguments:
           board: ndarray representation of the board
           player: the current player
           Return:
           int: -1 if the opponent can't win in only one move or the column where the opponent is going to win

               """
    # sets the opponent
    player2 = 0
    if player == BoardPiece(1):
        player2 = BoardPiece(2)
    else:
        player2 = BoardPiece(1)
    # checks if opponent has 3 pieces with space between
    if evaluate_curr_board(board, player2) > 1000:
        # find where opponent can make win
        for col in range(COL):
            tmp_board = board.copy()
            board_with_move = common.apply_player_action(board, col, player2, True)
            # when opponent wins return the column where the opponent is going to win
            if common.check_end_state(board_with_move, player2, col) == GameState.IS_WIN:
                board = tmp_board.copy()
                return col
            board = tmp_board.copy()

    return -1
