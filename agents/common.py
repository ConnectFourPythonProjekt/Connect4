from enum import Enum
from typing import Optional
import numpy as np
from typing import Callable, Tuple

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((6, 7), BoardPiece(0))



def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    """Player1 == X Player2 == O"""
    EndStr = '|==============|\n|0 1 2 3 4 5 6 |'
    for row in range(6):
        tmpString = '|'
        for cell in range(7):
            if board[row, cell] == PLAYER1:
                tmpString += 'X '
            elif board[row, cell] == PLAYER2:
                tmpString += 'O '
            else:
                tmpString += '  '
        tmpString += '|'
        EndStr = tmpString + '\n' + EndStr

    return '|==============|\n' + EndStr



def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    """Player1 == X Player2 == O"""
    BoardArr = np.zeros((6, 7), BoardPiece(0))
    splitStr = pp_board.split('\n')
    splitStr.pop(8)
    splitStr.pop(7)
    splitStr.pop(0)
    for row in range(6):
        rowStr = splitStr.pop(0)
        tmpRow = []
        for cell in range(1, 15, 2):
            if rowStr[cell] == 'O':
                tmpRow.append(PLAYER2)
            elif rowStr[cell] == 'X':
                tmpRow.append(PLAYER1)
            else:
                tmpRow.append(NO_PLAYER)
        BoardArr[5 - row] = np.asarray(tmpRow)

    return BoardArr



def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy == True:
        board_before = board.copy()
    for i in range(board.shape[0]):
        if board[i, action] == NO_PLAYER:
            board[i, action] = player
            break
        else:
            continue

    return board

def get_Last_Action_Col(board: np.ndarray, player: BoardPiece) -> np.ndarray:
    LastActions = np.ndarray(5,2)
    indRow = 5
    for col in range(6,-1,-1):
        for row in range(5,-1,-1):
            if board[row,col] == player:
                indRow +=1
                LastActions[col, 0] = indRow
                LastActions[col, 1] = col
            indRow -= 1


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    # is last action given
    # if last_action is None:
    #     return connected_four_without_LastAction(board, player)

    # with rec
    if last_action is None:
        for rec in range(7):
            check = connected_four(board, player, rec)
            if check:
                return True
            elif not check and rec == 6:
                return False
            else:
                continue

    BoardCol = (board[0:6, last_action]).T
    # find row of the last action
    # row = 0
    # i = 5
    # while i >= 0:
    #     if BoardCol[i] == player:
    #         row = i
    #         break
    #     i -= 1


    row = len(np.argwhere(board[:, last_action] == 0))
    row = board.shape[0] - row - 1
    # check horizontal
    BoardRow = board[row, 0:7].T
    c = 0
    for index in range(7):
        if BoardRow[index] == player:
            c += 1
            if c == 4:
                return True
        else:
            c = 0

    # check vertical
    count = 0
    for ind in range(6):
        if BoardCol[ind] == player:
            count += 1
            if count == 4:
                return True
        else:
            count = 0

    # check diagonal(main)
    diagList = np.diag(board, (last_action - row))
    counter = 0
    for value in diagList:
        if value == player:
            counter += 1
            if counter == 4:
                return True
        else:
            counter = 0

    # check diagonal(opposite)
    DiagList = np.diag(board[::-1], (last_action - (5 - row)))
    C = 0
    for val in DiagList:
        if val == player:
            C += 1
            if C == 4:
                return True
        else:
            C = 0
    return False


def connected_four_without_LastAction(board: np.ndarray, player: BoardPiece) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    # check horizontal
    for row in range(6):
        if np.count_nonzero(board[row, 0:7] == player) > 3:
            BoardRow = board[row, 0:7]
            c = 0
            for index in range(7):
                if BoardRow[index] == player:
                    c += 1
                    if c == 4:
                        return True
                else:
                    c = 0

    # check vertical
    for col in range(7):
        if np.count_nonzero(board[0:7, col] == player) > 3:
            BoardCol = board[0:7, col].T
            count = 0
            for ind in range(6):
                if BoardCol[ind] == player:
                    count += 1
                    if count == 4:
                        return True
                else:
                    count = 0

    # check diagonal(main)
    for index_diag in range(-2, 4):
        if np.count_nonzero(np.diag(board, index_diag) == player) > 3:
            diagList = np.diag(board, index_diag)
            counter = 0
            for value in diagList:
                if value == player:
                    counter += 1
                    if counter == 4:
                        return True
                else:
                    counter = 0

    # check diagonal(opposite)
    for index_opp_diag in range(-2, 4):
        if np.count_nonzero(np.diag(board[::-1], index_opp_diag) == player) > 3:
            DiagList = np.diag(board[::-1], index_opp_diag)
            C = 0
            for val in DiagList:
                if val == player:
                    C += 1
                    if C == 4:
                        return True
                else:
                    C = 0

    return False


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif np.count_nonzero(board == 0) < 1:
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING
