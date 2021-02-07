from typing import Optional, Tuple
import numpy as np

from agents import common
from agents.common import check_end_state, apply_player_action, GameState, BoardPiece, SavedState, PlayerAction
import random
import time

WIN = 1  # new_leaves player wins
LOST = -1  # new_leaves player loses
DRAW = 0
ROW = 6
COL = 7
PERIOD_OF_TIME = 3  # sec


class Node:
    def __init__(self, score=None, board_state=None, move=None, parent=None, player=None):
        self.UCB_score = score
        self.children = []
        self.board_state = board_state
        self.move = move
        self.parent = parent
        self.simulations = 0
        self.wins = 0
        self.player = player
        # self.done_moves = []

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
    tic = time.time()
    c = 1
    while True:
        best_node = selection(root, tree)
        new_node, updated_tree = expansion(best_node, tree)
        board_copy = board.copy()
        outcome = simulation(new_node, board_copy)
        root, final_tree = backpropagation(new_node, outcome, updated_tree)
        toc = time.time()
        toc_tic = toc - tic
        c += 1
        if toc_tic > PERIOD_OF_TIME: break
    print(c)
    tuc = time.time()

    print(f"Time after MCTR with loop in {tuc - tic:0.4f} sec")
    # find the best move
    win_rates = [[], []]
    for child in root.children:
        win_rate = child.wins / child.simulations
        win_rates[0].append(win_rate)
        win_rates[1].append(child)
    max_score = max(win_rates[0])
    max_score_index = win_rates[0].index(max_score)
    best_node = win_rates[1][max_score_index]
    toc2 = time.time()
    print(f"Time + best node move {toc2 - tic:0.4f} sec")
    print(best_node.move)
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
        node.UCB_score = upper_confidence_bound(node)
        best_score[0].append(node.UCB_score)
        best_score[1].append(node)
    max_score = max(best_score[0])
    max_score_index = best_score[0].index(max_score)

    return best_score[1][max_score_index]


def expansion(selected_node: Node, tree: Tree) -> Tuple[Node, Tree]:
    """
    The function expands the selected node and creates children node.
    Returns new node and updated tree with the newly created node
    """

    selected_node.add_node()
    new_node = selected_node.children[len(selected_node.children) - 1]
    tree.add_to_nodes(new_node)
    new_node.player = other_player(selected_node.player)
    return new_node, tree


def simulation(newly_created_node: Node, board: np.ndarray) -> int:
    """
    Called until the game is finished (there is a win or a draw)
    and return the result
    """

    opponent = other_player(newly_created_node.player)
    move = valid_move(board, opponent)
    newly_created_node.move = move

    board_copy = board.copy()
    newly_created_node.board_state = apply_player_action(board_copy, move, opponent, False)

    on_turn = newly_created_node.player
    while (check_end_state(board_copy, newly_created_node.player) == GameState.STILL_PLAYING and
           check_end_state(board_copy, opponent) == GameState.STILL_PLAYING):
        if valid_move(board_copy, on_turn) is None:
            break
        else:
            move = valid_move(board_copy, on_turn)

        # do moves
        if on_turn == newly_created_node.player:
            apply_player_action(board_copy, move, newly_created_node.player)
            on_turn = opponent
        else:
            apply_player_action(board_copy, move, opponent)
            on_turn = newly_created_node.player
        # game ends
        if check_end_state(board_copy, newly_created_node.player) == GameState.IS_WIN:
            return LOST  # newly_created nodes player wins
        if check_end_state(board_copy, opponent) == GameState.IS_WIN:
            return WIN  # newly_created nodes player lost
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


def valid_move(board: np.ndarray, player: BoardPiece) -> int:
    """
    Return a valid random move in the board
    """
    opponent = other_player(player)
    board_copy = board.copy()
    valid_moves_three = []
    block_opponent = []
    valid_moves_two = []
    valid_moves_one = []
    for col in range(board.shape[1]):
        if np.count_nonzero(board[:, col] == 0) > 0:  # check whether the column is full
            apply_player_action(board, col, player)
            position, mask = get_position_mask_bitmap(player, board)
            if connected_four(position):
                return col
            elif connected_three(position):
                valid_moves_three.append(col)
            elif connected_two(position):
                valid_moves_two.append(col)
            else:
                valid_moves_one.append(col)
            board = board_copy.copy()

            apply_player_action(board, col, opponent)
            position, mask = get_position_mask_bitmap(opponent, board)
            if connected_four(position):
                block_opponent.append(col)
        board = board_copy.copy()
    if len(valid_moves_one) == 0 and len(valid_moves_two) == 0 and len(valid_moves_three) == 0:
        return None
    else:
        if block_opponent:
            return random.choice(block_opponent)
        elif valid_moves_three:
            return random.choice(valid_moves_three)
        elif valid_moves_two:
            return random.choice(valid_moves_two)
        else:
            return random.choice(valid_moves_one)


def other_player(player: BoardPiece) -> BoardPiece:
    if player == BoardPiece(1):
        return BoardPiece(2)
    else:
        return BoardPiece(1)


def get_position_mask_bitmap(player: BoardPiece, board: np.ndarray) -> Tuple[int, int]:
    # board = node.board_state
    # player = other_player(node)
    position, mask = '', ''
    # Start with right-most column
    for j in range(6, -1, -1):
        # Add 0-bits to sentinel
        mask += '0'
        position += '0'
        # Start with bottom row
        for i in range(0, 6):
            mask += ['0', '1'][board[i, j] != 0]
            position += ['0', '1'][board[i, j] == player]
    return int(position, 2), int(mask, 2)


def connected_three(position: int):
    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 8):
        return True
    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 6):
        return True

    # Horizontal
    m = position & (position >> 7)
    if m & (m >> 7):
        return True

    # Vertical
    m = position & (position >> 1)
    if m & (m >> 1):
        return True

    # Nothing found
    return False


def connected_two(position: int):
    # Diagonal /
    if position & (position >> 8):
        return True
    # Diagonal \
    if position & (position >> 6):
        return True

    # Horizontal and Vertical
    if position & (position >> 7):
        return True
    # Nothing found
    return False


def connected_four(position):
    # Horizontal check
    m = position & (position >> 7)
    if m & (m >> 14):
        return True
    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 12):
        return True
    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 16):
        return True
    # Vertical
    m = position & (position >> 1)
    if m & (m >> 2):
        return True
    # Nothing found
    return False
