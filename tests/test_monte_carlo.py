from agents.agents_montecarlo.monte_carlo import Node, choose_move
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
    assert not root == child_note
    assert child_note.parent == root
    assert child_note.score == 23
    assert child_note.move == 4
    assert root.parent is None


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
    child.add_node()
    child2 = child.children[0]
    child.add_node()
    child3 = child.children[1]

    assert child.children[0] == child2
    assert child2.parent == child
    assert child.parent == root
    assert root.children[0] == child
    assert child.children[1] == child3
    assert not child.children[0] == child3
    assert not root == child2
    assert not child2.parent == root
    assert not root == child
    assert not child == child2


def test_choose_move():
    root = Node()
    root.wins = 5
    root.simulations = 11
    root.add_node()
    root.add_node()
    root.add_node()
    child1 = root.children[0]
    child2 = root.children[1]
    child3 = root.children[2]

    child1.wins = 2
    child1.simulations = 5
    child2.wins = 2
    child1.move = 2
    child2.simulations = 3
    child2.move = 0
    child3.wins = 0
    child3.simulations = 1
    child3.move = 5

    assert choose_move(root) == child2.move
    assert not choose_move(root) == child1
    assert not choose_move(root) == child3
