from agents.agents_minimax.minimax import generate_move_minimax
from agents.common import apply_player_action, initialize_game_state, BoardPiece


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

    # board2 = initialize_game_state()
    # apply_player_action(board2, 3, BoardPiece(2), False)
    # apply_player_action(board2, 3, BoardPiece(1), False)
    # apply_player_action(board2, 0, BoardPiece(2), False)
    # action2, save_state = generate_move_minimax(board2, BoardPiece(1), None)
    # assert action2 == 2