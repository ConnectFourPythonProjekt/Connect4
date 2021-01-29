from typing import Optional, Tuple
import numpy as np
from agents.common import check_end_state, apply_player_action, GameState, BoardPiece, SavedState, PlayerAction
import random

LOOP = 150
WIN = 1
LOST = -1
DRAW = 0


class Node:
    def __init__(self, score=None, board_state=None, move=None, parent=None, player=None):
        self.score = score
        self.children = []
        self.board_state = board_state
        self.move = move
        self.parent = parent
        self.simulations = 0
        self.wins = 0
        self.player = player

    def add_node(self):
        new_node = Node()
        self.children.append(new_node)
        new_node.parent = self

    def __repr__(self):
        return f'Node(parent={self.parent})'


class Tree:
    def __init__(self, root: Node):
        self.nodes = []
        self.root = root

    def add_to_nodes(self, node: Node):
        self.nodes.append(node)


def generate_move_montecarlo(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    root = Node(board_state=board, player=player)
    tree = Tree(root)
    move = MCTS(root, board, tree)
    return move, saved_state


def MCTS(root: Node, board: np.ndarray, tree: Tree) -> int:
    for i in range(LOOP):
        best_node = selection(root, tree)
        new_node, updated_tree = expansion(best_node, tree)
        board_copy = board.copy()
        outcome = simulation(new_node, board_copy, new_node.player)
        root, final_tree = backpropagation(new_node, outcome, updated_tree)

    win_rates = [[], []]
    for child in root.children:
        win_rate = child.wins / child.simulations
        win_rates[0].append(win_rate)
        win_rates[1].append(child)
    max_score = max(win_rates[0])
    max_score_index = win_rates[0].index(max_score)
    best_node = win_rates[1][max_score_index]
    return best_node.move


def upper_confidence_bound(current_node: Node) -> float:
    w = current_node.wins
    s = current_node.simulations
    sp = current_node.parent.simulations
    c = np.sqrt(2)

    return (w / s) + c * (np.sqrt((np.log(sp)) / s))


def selection(node: Node, tree: Tree) -> Node:
    """
    Returns the node that has the highest possibility of winning (UCB1)
    """
    if (node.parent is None) and (not node.children):
        return node

    best_score = [[], []]
    for node in tree.nodes:
        node.score = upper_confidence_bound(node)
        best_score[0].append(node.score)
        best_score[1].append(node)
    max_score = max(best_score[0])
    max_score_index = best_score[0].index(max_score)

    return best_score[1][max_score_index]


def expansion(selected_node: Node, tree: Tree) -> Tuple[Node, Tree]:
    """
    The function expands the selected node and creates many children nodes.
    Returns the tree with the newly created node
    and the newly created node
    Called by simulation
    """
    if selected_node.player == BoardPiece(1):
        opponent = BoardPiece(2)
    else:
        opponent = BoardPiece(1)

    selected_node.add_node()
    new_node = selected_node.children[len(selected_node.children) - 1]
    tree.add_to_nodes(new_node)
    new_node.player = opponent
    return new_node, tree


def simulation(newly_created_node: Node, board: np.ndarray, player: BoardPiece) -> int:
    """
    Recursivly called until the game is finished and a winner emerges
    updates the score of the new node
    """
    on_turn = player

    if newly_created_node.player == BoardPiece(1):
        opponent = BoardPiece(2)
    else:
        opponent = BoardPiece(1)

    move = valid_move(board)
    newly_created_node.move = move
    newly_created_node.board_state = board
    while (check_end_state(board, newly_created_node.player) == GameState.STILL_PLAYING and
           check_end_state(board, opponent) == GameState.STILL_PLAYING):
        # do moves
        if on_turn == newly_created_node.player:
            apply_player_action(board, move, newly_created_node.player)
            on_turn = opponent
        else:
            apply_player_action(board, move, opponent)
            on_turn = newly_created_node.player
        # game ends
        if check_end_state(board, newly_created_node.player) == GameState.IS_WIN:
            return WIN
        if check_end_state(board, opponent) == GameState.IS_WIN:
            return LOST
        move = valid_move(board)

    return DRAW


def backpropagation(newly_created_node: Node, outcome: int, tree: Tree) -> Tuple[Node, Tree]:
    """
    Update the parent scores one by one by going up the tree
    Returns the tree with the updated scores
    """

    newly_created_node.simulations += 1  # new node simulations
# TODO: trqbwa da se dobawqt win ako e dr ugracha l186 i l175
    if outcome == LOST:
        newly_created_node.wins += 1
        node = newly_created_node.parent
        while node.parent is not None:
            if newly_created_node.player == node.player:
                node.wins += 1
            node.simulations += 1
            node = node.parent

    elif outcome == WIN:
        newly_created_node.wins += 1
        node = newly_created_node.parent
        while node.parent is not None:
            if newly_created_node.player != node.player:
                node.wins += 1
            node.simulations += 1
            node = node.parent
    else:
        node = newly_created_node.parent
        while node.parent is not None:
            node.simulations += 1
            node = node.parent

    node.simulations += 1  # root simulations

    return node, tree


def valid_move(board) -> int:
    valid_moves = []  # list with columns, where it can be played
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:, col] == 0) > 0:  # check whether the column is full
            valid_moves.append(col)

    return random.choice(valid_moves)
