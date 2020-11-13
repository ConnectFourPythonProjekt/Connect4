import numpy as np
from agents.common import BoardPiece, NO_PLAYER


def test_initialize_game_state():
    from agents.common import initialize_game_state
    ret = initialize_game_state()
    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)

def test_apply_player_action():
    from agents.common import initialize_game_state
    from agents.common import apply_player_action
    board = initialize_game_state()
    board[0,1] = BoardPiece(1)
    board[0,2] = BoardPiece(2)
    board[1,1] = BoardPiece(1)
    ret = apply_player_action(board,2,BoardPiece(2),False)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert isinstance(ret,np.ndarray)
    assert ret[1,2] == BoardPiece(2)
    board = initialize_game_state()
    ret = apply_player_action(board, 2, BoardPiece(2), False)
    assert ret[0, 2] == BoardPiece(2)
    ret = apply_player_action(board, 2, BoardPiece(1), False)
    assert ret[1, 2] == BoardPiece(1)
    ret = apply_player_action(board, 3, BoardPiece(2), False)
    assert ret[0, 3] == BoardPiece(2)
    ret = apply_player_action(board, 3, BoardPiece(1), False)
    assert ret[1, 3] == BoardPiece(1)
    ret = apply_player_action(board, 2, BoardPiece(2), False)
    assert ret[2, 2] == BoardPiece(2)
    assert isinstance(ret,np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)

def test_connected_four():
    from agents.common import initialize_game_state
    from agents.common import apply_player_action
    from agents.common import connected_four
    board = initialize_game_state()
    # True tests
    # vertical
    apply_player_action(board, 2, BoardPiece(2), False)

    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    ret = connected_four(board,BoardPiece(1),2)
    assert isinstance(ret,bool)
    assert ret == True
    # horizontal
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(1), False)

    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 3, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 5, BoardPiece(1), False)
    ret = connected_four(board,1,5)
    assert isinstance(ret, bool)
    assert ret == True
    # left right diagonal
    board = initialize_game_state()
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 0, BoardPiece(2), False)
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 1, BoardPiece(2), False)
    apply_player_action(board, 1, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 5, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 3, BoardPiece(1), False)
    ret = connected_four(board, 1,3)
    assert isinstance(ret, bool)
    assert ret == True
    # right left diagonal
    board = initialize_game_state()
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 0, BoardPiece(2), False)
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 1, BoardPiece(2), False)
    apply_player_action(board, 1, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 5, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 1, BoardPiece(2), False)
    apply_player_action(board, 3, BoardPiece(1), False)
    apply_player_action(board, 0, BoardPiece(2), False)

    ret = connected_four(board, 2, 0)
    assert isinstance(ret, bool)
    assert ret == True
    #false tests
    # vertical
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(2), False)

    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    ret = connected_four(board, BoardPiece(2), 3)
    assert ret == False
    # horizontal
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(1), False)

    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 3, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    ret = connected_four(board, 2, 2)
    assert isinstance(ret, bool)
    assert ret == False
    # left right diagonal
    board = initialize_game_state()
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 0, BoardPiece(2), False)
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 1, BoardPiece(2), False)
    apply_player_action(board, 1, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 5, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    ret = connected_four(board, BoardPiece(1), 4)

    assert ret == False
    # right left diagonal
    board = initialize_game_state()
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 0, BoardPiece(2), False)
    apply_player_action(board, 0, BoardPiece(1), False)
    apply_player_action(board, 1, BoardPiece(2), False)
    apply_player_action(board, 1, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 5, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 1, BoardPiece(2), False)

    ret = connected_four(board, 2, 1)
    assert isinstance(ret, bool)
    assert ret == False
def test_check_end_state():
    from agents.common import check_end_state
    from agents.common import apply_player_action
    from agents.common import initialize_game_state
    from agents.common import GameState
    from agents.common import pretty_print_board
    # test 'is win'
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 3, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 4, BoardPiece(1), False)
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 5, BoardPiece(1), False)

    ret = check_end_state(board,BoardPiece(1),5)
    assert isinstance(ret, GameState)
    assert ret == GameState.IS_WIN
    # test still playing
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 3, BoardPiece(1), False)
    ret = check_end_state(board,1,3)
    assert ret == GameState.STILL_PLAYING
    #test is draw
    board[:,0] = BoardPiece(1)
    board[:,1:3] = BoardPiece(2)
    board[:, 3:5] = BoardPiece(1)
    board[:,5:7] = BoardPiece(2)
    board[3:5,:] = BoardPiece(1)
    board[1,:] = BoardPiece(2)
    ret = check_end_state(board,2,5)
    assert ret == GameState.IS_DRAW











