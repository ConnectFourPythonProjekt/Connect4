import numpy as np
from agents.common import check_end_state, apply_player_action, GameState, BoardPiece
import random


class Node():
    def __init__(self, score=None, board_state=None, move=None, parent=None):
        self.score = score
        self.children = []
        self.board_state = board_state
        self.move = move
        self.parent = parent
        self.simulations = 0
        self.wins = 0
        self.player = BoardPiece

    def add_node(self):
        new_node = Node(parent=self)
        self.children.append(new_node)


def upper_confidence_bound(current_node: Node) -> float:
    w = current_node.wins
    s = current_node.simulations
    sp = current_node.parent.simulations
    c = np.sqrt(2)

    return (w / s) + c * (np.sqrt((np.log(sp)) / s))


def selection(root: Node):
    """
    Returns the node that has the highest possibility of winning (UCB1)
    Called by expansion
    """
    best_score = [[], []]
    if not root.children:
        return root
    for child in root.children:
        child.score = upper_confidence_bound(child)
        best_score[0].append(child.score)
        best_score[1].append(child)
    max_score = max(best_score[0])
    max_score_index = best_score[0].index(max_score)
    best_node = best_score[1][max_score_index]
    selection(best_node)
    expansion(best_node)


def expansion(selected_node: Node) -> Node:
    """
    The function expands the selected node and creates many children nodes.
    Returns the tree with the newly created node
    and the newly created node
    Called by simulation
    """
    selected_node.add_node()
    child = selected_node.children
    simulation(child[0])


def valid_moves(board) -> int:
    valid_moves = []  # list with columns, where it can be played
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:, col] == 0) > 0:  # check whether the column is full
            valid_moves.append(col)

    return random.choice(valid_moves)


def simulation(newly_created_node: Node):
    """
    Recursivly called until the game is finished and a winner emerges
    updates the score of the new node
    """
    on_turn = newly_created_node.player
    board = newly_created_node.board_state
    newly_created_node.simulations += 1
    counter = 0
    winner = False
    if newly_created_node.player == BoardPiece(1):
        opponent = BoardPiece(2)
    else:
        opponent = BoardPiece(1)
    while (check_end_state(board, newly_created_node.player) == GameState.STILL_PLAYING and
           check_end_state(board, opponent) == GameState.STILL_PLAYING):

        if on_turn == newly_created_node.player:
            move = valid_moves(board)
            apply_player_action(board, move, newly_created_node.player)
            if counter == 0:
                newly_created_node.move = move
                counter = 1
            on_turn = opponent
        else:
            apply_player_action(board, valid_moves(board), opponent)
            on_turn = newly_created_node.player
        if check_end_state(board,newly_created_node.player) == GameState.IS_WIN:
            newly_created_node.wins += 1
            winner = True
    backpropagation(newly_created_node,winner)


def backpropagation(newly_created_node: Node, winner : bool):
    """
    Update the parent scores one by one by going up the tree
    Returns the tree with the updated scores
    """
    parent = newly_created_node.parent
    if parent is not None:
        if winner:
            parent.wins += 1
        parent.simulations += 1
        backpropagation(parent,winner)
    else:
        





