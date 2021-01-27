from agents.agents_montecarlo.monte_carlo import Node
from agents.agents_montecarlo import monte_carlo
from agents.common import initialize_game_state, apply_player_action


def test_node():
    board = initialize_game_state()
    board[0, 0:7] = [0, 1, 1, 2, 1, 0, 0]
    board[1, 0:7] = [0, 0, 0, 2, 0, 0, 0]
    board[2, 0:7] = [0, 0, 0, 1, 0, 0, 0]
    board[3, 0:7] = [0, 0, 0, 2, 0, 0, 0]
    board[4, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board[5, 0:7] = [0, 0, 0, 0, 0, 0, 0]

    board_child = apply_player_action(board, 4, 2)
    root = Node(49, board)
    child_note = Node(23, board_child, 4, root)
    assert child_note.parent == root
    assert child_note.score == 23
    assert child_note.move == 4


def test_add_node():
    board = initialize_game_state()
    board[0, 0:7] = [0, 1, 1, 2, 1, 0, 0]
    board[1, 0:7] = [0, 0, 0, 2, 0, 0, 0]
    board[2, 0:7] = [0, 0, 0, 1, 0, 0, 0]
    board[3, 0:7] = [0, 0, 0, 2, 0, 0, 0]
    board[4, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    board[5, 0:7] = [0, 0, 0, 0, 0, 0, 0]
    root = Node(12, board)
    root.add_node()
    child = root.children[0]
    f = 4



    # assert root.children.pop(0) == child
    # assert child.parent == root



