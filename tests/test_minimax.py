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
    board = initialize_game_state()
    apply_player_action(board, 2, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    apply_player_action(board, 2, BoardPiece(1), False)
    apply_player_action(board, 3, BoardPiece(2), False)
    action1, save_state = generate_move_minimax(board, BoardPiece(2), None)
    assert action1 == 2
