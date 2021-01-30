from typing import Optional, Tuple
import numpy as np
from agents.common import check_end_state, apply_player_action, GameState, BoardPiece, SavedState, PlayerAction
import random

LOOP = 150
WIN = 1  # new_leaves player wins
LOST = -1  # new_leaves player loses
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
    for i in range(LOOP):  # number of playouts
        best_node = selection(root, tree)
        new_node, updated_tree = expansion(best_node, tree)
        board_copy = board.copy()
        outcome = simulation(new_node, board_copy, new_node.player)
        root, final_tree = backpropagation(new_node, outcome, updated_tree)

    # find the best move
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
    # tree with only root we select the root
    if (node.parent is None) and (not node.children):
        return node

    # finding the best child with UCB1
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
    The function expands the selected node and creates children node.
    Returns new node and updated tree with the newly created node
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
    Recursively called until the game is finished (there is a win or a draw)
    and return the result
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
            return WIN  # newly_created nodes player wins
        if check_end_state(board, opponent) == GameState.IS_WIN:
            return LOST  # newly_created nodes player lost
        move = valid_move(board)

    return DRAW


def backpropagation(newly_created_node: Node, outcome: int, tree: Tree) -> Tuple[Node, Tree]:
    """
    Update nodes scores one by one by going up the tree until the root
    Returns the root and tree with the updated scores
    """

    newly_created_node.simulations += 1  # new node simulations

    if outcome == LOST:
        newly_created_node.wins += 1
        node = newly_created_node.parent
        while node.parent is not None:
            if newly_created_node.player == node.player:
                node.wins += 1
            node.simulations += 1
            node = node.parent
        node.simulations += 1
        if newly_created_node.player == node.player:
            node.wins += 1

    elif outcome == WIN:
        node = newly_created_node.parent
        while node.parent is not None:
            if newly_created_node.player != node.player:
                node.wins += 1
            node.simulations += 1
            node = node.parent
        node.simulations += 1
        if newly_created_node.player != node.player:
            node.wins += 1
    else:
        node = newly_created_node.parent
        while node.parent is not None:
            node.simulations += 1
            node = node.parent
        node.simulations += 1

    return node, tree


def valid_move(board) -> int:
    """
    Return a valid random move in the board
    """
    valid_moves = []  # list with columns, where it can be played
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:, col] == 0) > 0:  # check whether the column is full
            valid_moves.append(col)

    return random.choice(valid_moves)
