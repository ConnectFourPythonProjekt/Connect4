from typing import Optional, Tuple
import numpy as np
from agents.common import check_end_state, apply_player_action, GameState, BoardPiece, SavedState, PlayerAction
import random

LOOP = 4

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


def generate_move_montecarlo(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    root = Node(board_state=board, player=player)
    # MCTR
    selection(root)
    move = choose_move(root)

    return move, saved_state


def choose_move(root: Node) -> int:
    win_rates = [[], []]
    if not root.children:
        return root
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


def selection(node: Node):
    """
    Returns the node that has the highest possibility of winning (UCB1)
    """
    best_score = [[], []]
    if not node.children:
        expansion(node)
    for child in node.children:
        child.score = upper_confidence_bound(child)
        best_score[0].append(child.score)
        best_score[1].append(child)
    max_score = max(best_score[0])
    max_score_index = best_score[0].index(max_score)
    best_node = best_score[1][max_score_index]

    selection(best_node)
    expansion(best_node)

def expansion(selected_node: Node):
    """
    The function expands the selected node and creates many children nodes.
    Returns the tree with the newly created node
    and the newly created node
    Called by simulation
    """
    selected_node.add_node()
    child = selected_node.children
    simulation(child[0])

def simulation(newly_created_node: Node):
    """
    Recursivly called until the game is finished and a winner emerges
    updates the score of the new node
    """
    on_turn = newly_created_node.player
    board = newly_created_node.board_state

    if newly_created_node.player == BoardPiece(1):
        opponent = BoardPiece(2)
    else:
        opponent = BoardPiece(1)

    move = valid_move(board)
    newly_created_node.move = move
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
            backpropagation(newly_created_node, newly_created_node.player)
        if check_end_state(board, opponent) == GameState.IS_WIN:
            backpropagation(newly_created_node, opponent)
        move = valid_move(board)

    backpropagation(newly_created_node)


def backpropagation(newly_created_node: Node, winner: None):
    """
    Update the parent scores one by one by going up the tree
    Returns the tree with the updated scores
    """
    newly_created_node.simulations += 1
    if winner is not None:
        if (winner is BoardPiece(1) and newly_created_node.player is BoardPiece(2)) or (
                winner is BoardPiece(2) and newly_created_node.player is BoardPiece(1)):
            newly_created_node.wins += 1
    if newly_created_node.parent is not None:
        backpropagation(newly_created_node.parent, winner)
    else:
        choose_move(newly_created_node)


def valid_move(board) -> int:
    valid_moves = []  # list with columns, where it can be played
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:, col] == 0) > 0:  # check whether the column is full
            valid_moves.append(col)

    return random.choice(valid_moves)