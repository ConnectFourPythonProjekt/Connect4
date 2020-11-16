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


GenMove = Callable[[np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
                   Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).

    Arguments:
        ---

    Return:
        np.ndarray: initialized board, filled with zeros
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

    Player1 == X Player 2 == O

    Arguments:
        board: ndarray representation of the board

    Return:
        str: human readable string representation of the board
    """

    EndStr = '|==============|\n|0 1 2 3 4 5 6 |'
    # iteration through the board and string construction
    for row in range(6):
        tmpString = '|'     # begin of the row
        for cell in range(7):
            if board[row, cell] == PLAYER1:
                tmpString += 'X '
            elif board[row, cell] == PLAYER2:
                tmpString += 'O '
            else:
                tmpString += '  '
        tmpString += '|'        # end of the row
        EndStr = tmpString + '\n' + EndStr  # works as the stack principle and pushes the next row

    return '|==============|\n' + EndStr

def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.

    Player1 == X Player2 == O

    Arguments:
        pp_board: human readable string representation of the board

    Return:
        np.ndarray: ndarray representation of the board
    """

    BoardArr = np.zeros((6, 7), BoardPiece(0))
    splitStr = pp_board.split('\n')
    splitStr.pop(8)  # | pops the rows where there is
    splitStr.pop(7)  # | |==============| or
    splitStr.pop(0)  # | |0 1 2 3 4 5 6 |
    for row in range(5,-1,-1):
        rowStr = splitStr.pop(0)
        tmpRow = []  # stores the string characters converted to board pieces
        for cell in range(1, 15, 2):    # index 0 of each substring is | and every cell contains board piece and a space
            if rowStr[cell] == 'O':
                tmpRow.append(PLAYER2)
            elif rowStr[cell] == 'X':
                tmpRow.append(PLAYER1)
            else:
                tmpRow.append(NO_PLAYER)

        BoardArr[row] = np.asarray(tmpRow)  # works as the stack principle and pushes the next row

    return BoardArr


# TODO: trqbwa ni copy test
def apply_player_action(board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.

    Arguments:
        board: ndarray representation of the board
        action: column, where the player wants to have a piece
        player: the player whose turn it is
        copy: board copy if needed

    Return:
        np.ndarray: ndarray representation of the board including the last action

    """
    if copy:
        board_before = board.copy()
    for i in range(board.shape[0]):
        if board[i, action] == NO_PLAYER:
            board[i, action] = player
            break
    return board


def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.

    Arguments:
        board: ndarray representation of the board
        player: a player whose victory is being checked
        last_action: column, where the last action was

    Return:
        bool: True, when current player wins, False, when game isn't won by player
    """

    # checks if there is last action provided
    # if it's not provided iterate through the whole board and check if the player won
    if last_action is None:
        for column in range(7):
            check = connected_four(board, player, column)
            if check:
                return True
            if not check and column == 6:
                return False  # there is no win on the board and we already checked all columns

    # finds free row in this column
    row = len(np.argwhere(board[:, last_action] == 0))  # counts zeros in column
    row = board.shape[0] - row - 1      # the row of the last action

    # checks horizontal
    BoardRow = board[row, 0:7].T  # gets whole row of last action
    if four_in_a_row(BoardRow, player):
        return True

    # checks vertical
    BoardCol = (board[0:6, last_action]).T  # gets whole column of last action
    if four_in_a_row(BoardCol, player):
        return True

    # checks diagonal(main)
    diagList = np.diag(board, (last_action - row))  # gets (left over right down) diagonal of last action
    if four_in_a_row(diagList, player):
        return True

    # checks diagonal(opposite)
    DiagList = np.diag(board[::-1], (last_action - (5 - row)))  # gets (left down right over) diagonal of last action
    if four_in_a_row(DiagList, player):
        return True

    return False


def four_in_a_row(array: np.ndarray, player: BoardPiece) -> bool:
    """
        Counts how many adjacent pieces equal to `player` does the vertical/horizontal/diagonal contains.
        If they are four returns True, otherwise False.

        Arguments:
            array: ndarray to be checked if contains four adjacent pieces equal to `player`
            player: a player whose victory is being checked

        Return:
            bool: True, when current player wins, False, when game isn't won by player
        """
    counter = 0
    for index in array:
        if index == player:
            counter += 1
            if counter == 4:
                return True
        else:
            counter = 0
    return False


def check_end_state(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?

     Arguments:
        board: ndarray representation of the board
        player: a player whose victory is being checked
        last_action: column, where the last action was

    Return:
        GameState: current game state
    """

    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    elif np.count_nonzero(board == 0) < 1:
        return GameState.IS_DRAW
    return GameState.STILL_PLAYING
