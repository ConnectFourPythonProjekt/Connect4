from agents.agents_montecarlo.monte_carlo import Node, Tree, selection, expansion, simulation, backpropagation, generate_move_montecarlo \

from agents.common import initialize_game_state, apply_player_action, BoardPiece


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


def test_selection():
    root = Node()
    tree = Tree(root)
    root.wins = 5
    root.simulations = 10
    root.add_node()
    root.add_node()
    root.add_node()

    child1 = root.children[0]
    child2 = root.children[1]
    child3 = root.children[2]
    tree.add_to_nodes(child1)
    tree.add_to_nodes(child2)
    tree.add_to_nodes(child3)

    child1.wins = 2
    child1.simulations = 5
    child2.wins = 2
    child1.move = 2
    child2.simulations = 3
    child2.move = 0
    child3.wins = 0
    child3.simulations = 1
    child3.move = 5

    assert selection(root, tree) == child3

    root1 = Node()
    tree1 = Tree(root1)

    assert selection(root1, tree1) == root1


def test_expansion():
    root = Node()
    tree = Tree(root)
    root.add_node()
    selected_node = root.children[0]
    tree.add_to_nodes(selected_node)
    new_node, new_tree = expansion(selected_node, tree)
    assert new_tree.nodes[0] == selected_node
    assert new_tree.nodes[1] == new_node
    assert selected_node.children[0] == new_node
    assert new_node.parent == selected_node
    assert not selected_node.player == new_node.player
#
#
# def test_backpropagation():
#     # leaf player wins
#     root_tree = Node(player=BoardPiece(1))
#     tree = Tree(root_tree)
#
#     root_tree.add_node()
#     child1 = root_tree.children[0]
#
#     child1.add_node()
#     child2 = child1.children[0]
#
#     child2.add_node()
#     leaf = child2.children[0]
#
#     tree.add_to_nodes(child1)
#     tree.add_to_nodes(child2)
#     tree.add_to_nodes(leaf)
#
#     root_tree.simulations = 8
#     root_tree.wins = 5
#
#     child1.simulations = 5
#     child1.wins = 3
#     child1.player = 2
#
#     child2.simulations = 3
#     child2.wins = 2
#     child2.player = 1
#
#     leaf.simulations = 1
#     leaf.wins = 1
#     leaf.player = 2
#
#     new_node, tree = backpropagation(leaf, 1, tree)
#
#     assert leaf.simulations == 2
#     assert leaf.wins == 1
#
#     assert child2.wins == 3
#     assert child2.simulations == 4
#
#     assert child1.wins == 3
#     assert child1.simulations == 6
#
#     assert root_tree.wins == 6
#     assert root_tree.simulations == 9
#
#     # leaf player loses
#     root_tree1 = Node(player=BoardPiece(1))
#     tree1 = Tree(root_tree1)
#
#     root_tree1.add_node()
#     child11 = root_tree1.children[0]
#
#     child11.add_node()
#     child21 = child11.children[0]
#
#     child21.add_node()
#     leaf1 = child21.children[0]
#
#     tree1.add_to_nodes(child11)
#     tree1.add_to_nodes(child21)
#     tree1.add_to_nodes(leaf1)
#
#     root_tree1.simulations = 8
#     root_tree1.wins = 5
#
#     child11.simulations = 5
#     child11.wins = 3
#     child11.player = 2
#
#     child21.simulations = 3
#     child21.wins = 2
#     child21.player = 1
#
#     leaf1.simulations = 1
#     leaf1.wins = 1
#     leaf1.player = 2
#
#     new_node1, tree1 = backpropagation(leaf1, -1, tree1)
#
#     assert leaf1.simulations == 2
#     assert leaf1.wins == 2
#
#     assert child21.wins == 2
#     assert child21.simulations == 4
#
#     assert child11.wins == 4
#     assert child11.simulations == 6
#
#     assert root_tree1.wins == 5
#     assert root_tree1.simulations == 9
#
#     # is a draw
#     root_tree11 = Node(player=BoardPiece(1))
#     tree11 = Tree(root_tree11)
#
#     root_tree11.add_node()
#     child111 = root_tree11.children[0]
#
#     child111.add_node()
#     child211 = child111.children[0]
#
#     child211.add_node()
#     leaf11 = child211.children[0]
#
#     tree11.add_to_nodes(child111)
#     tree11.add_to_nodes(child211)
#     tree11.add_to_nodes(leaf11)
#
#     root_tree11.simulations = 8
#     root_tree11.wins = 5
#
#     child111.simulations = 5
#     child111.wins = 3
#     child111.player = 2
#
#     child211.simulations = 3
#     child211.wins = 2
#     child211.player = 1
#
#     leaf11.simulations = 1
#     leaf11.wins = 1
#     leaf11.player = 2
#
#     new_node11, tree11 = backpropagation(leaf11, 0, tree11)
#
#     assert leaf11.simulations == 2
#     assert leaf11.wins == 1
#
#     assert child211.wins == 2
#     assert child211.simulations == 4
#
#     assert child111.wins == 3
#     assert child111.simulations == 6
#
#     assert root_tree11.wins == 5
#     assert root_tree11.simulations == 9



