from agents.agents_minimax.minimax import generate_move_minimax
from agents.common import apply_player_action, initialize_game_state, BoardPiece,pretty_print_board


def test_generate_move_minimax():
    # win on the first move
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    action, save_state = generate_move_minimax(board, BoardPiece(1), None)
    assert action == 2

    # block opponent
    board1 = initialize_game_state()
    apply_player_action(board1, 2, BoardPiece(2), False)
    apply_player_action(board1, 0, BoardPiece(1), False)
    apply_player_action(board1, 3, BoardPiece(2), False)
    apply_player_action(board1, 2, BoardPiece(1), False)
    apply_player_action(board1, 3, BoardPiece(2), False)
    apply_player_action(board1, 0, BoardPiece(1), False)
    apply_player_action(board1, 3, BoardPiece(2), False)

    action1, save_state = generate_move_minimax(board1, BoardPiece(1), None)
    assert action1 == 3

    # with alpha-betta
    board2 = initialize_game_state()
    board2[0, 0:7] = [0, 1, 1, 2, 1, 0, 0]
    board2[1, 0:7] = [0, 0, 0, 2, 0, 0, 0]
    board2[2, 0:7] = [0, 0, 0, 1, 0, 0, 0]
    board2[3, 0:7] = [0, 0, 0, 2, 0, 0, 0]
    board2[4, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board2[5, 0:7] = [0, 0, 0, 0, 0, 0, 0]

    action2, save_state = generate_move_minimax(board2, BoardPiece(1), None)
    assert action2 == 2

    board3 = initialize_game_state()
    board3[0, 0:7] = [1, 0, 2, 1, 2, 0, 1]
    board3[1, 0:7] = [0, 0, 1, 2, 0, 0, 0]
    board3[2, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board3[3, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board3[4, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board3[5, 0:7] = [0, 0, 0, 0, 0, 0, 0]

    action3, save_state = generate_move_minimax(board3, BoardPiece(2), False)
    assert action3 == 2

    board4 = initialize_game_state()
    board4[0, 0:7] = [1, 2, 2, 1, 1, 2, 2]
    board4[1, 0:7] = [1, 2, 2, 2, 1, 1, 2]
    board4[2, 0:7] = [2, 1, 1, 1, 2, 2, 1]
    board4[3, 0:7] = [1, 1, 2, 1, 1, 2, 2]
    board4[4, 0:7] = [2, 1, 2, 2, 1, 1, 1]
    board4[5, 0:7] = [1, 2, 2, 1, 2, 0, 2]
    action4, save_state = generate_move_minimax(board4, BoardPiece(2), False)
    assert action4 == 5