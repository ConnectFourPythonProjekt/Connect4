from typing import Optional, Tuple
import numpy as np
from agents.common import check_end_state, apply_player_action, GameState, BoardPiece, SavedState, PlayerAction
import random
import time

WIN = 1
DRAW = 0
LOST = -1
FULL = -2  # the board is full
PERIOD_OF_TIME = 3  # in sec; set the timer


class Node:
    def __init__(self, board_state=None, move=None, parent=None, player=None):
        self.children = []
        self.board_state = board_state
        self.move = move
        self.parent = parent
        self.simulations = 0
        self.wins = 0
        self.player = player
        self.value = 0
        self.explored = False  # when True node can't be selected in selection

    def add_node(self):
        new_node = Node()
        self.children.append(new_node)
        new_node.parent = self

    def __repr__(self):
        return f' Move: {self.move} Wins: {self.wins} Simulation: {self.simulations} value: {self.value}'


class Tree:
    def __init__(self, root: Node):
        self.nodes = []
        self.root = root

    def add_to_nodes(self, node: Node):
        self.nodes.append(node)


def generate_move_montecarlo(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) -> Tuple[
    PlayerAction, Optional[SavedState]]:
    """
       Choose a move based on Monte Carlo tree search algorithm

       Arguments:
       board: ndarray representation of the board
       player: whether agent plays with X (Player1) or O (Player2)
       saved_state: computation that it could reuse for future moves

       Return:
           Tuple[PlayerAction, SavedState]: returns the column, where the agent will play and agent computations
       """
    root = Node(board_state=board, player=player)
    tree = Tree(root)
    move = MCTS(root, board, tree)
    return move, saved_state


def MCTS(root: Node, board: np.ndarray, tree: Tree) -> int:
    """
    Monte Carlo tree search algorithm executing for fixed amount of time
    Arguments:
        root: root of the tree, starting node
        board: ndarray representation of the board
        tree: contains all the nodes

    Return:
        int: return the best move
    """
    tic = time.time()  # start timer
    root.board_state = board

    # execute algorithm until time is out
    while True:
        best_node = selection(root, tree)
        new_node, updated_tree = expansion(best_node, tree)
        outcome = simulation(new_node, new_node.parent.board_state.copy())
        if outcome != FULL:
            root, final_tree = backpropagation(new_node, outcome, updated_tree)
        if time.time() - tic > PERIOD_OF_TIME: break  # out of time

    # find child with best move
    win_rates = {child.move: child.value for child in root.children}
    return max(win_rates, key=win_rates.get)


def upper_confidence_bound(current_node: Node) -> float:
    """
    Upper confidence bound formula
    Arguments:
        current_node: node for which we will calculate UCB score
    Returns:
        float: UCB score
    """
    w = current_node.wins
    s = current_node.simulations
    sp = current_node.parent.simulations
    c = np.sqrt(2)

    return (w / s) + c * (np.sqrt((np.log(sp)) / s))


def selection(node: Node, tree: Tree) -> Node:
    """
    Returns the node that has the highest possibility of winning (UCB1)

    Arguments:
        node: root node
        tree: contains all the nodes
    Returns:
        Node: return the selected node
    """

    # first select all 7 children of the root
    if (node.parent is None) and (len(node.children) < 7):
        return node

    # finding best child with UCB
    best_score = {
        item: upper_confidence_bound(item)
        for item in tree.nodes
        if not item.explored
    }
    return max(best_score, key=best_score.get)


def expansion(selected_node: Node, tree: Tree) -> Tuple[Node, Tree]:
    """
    The function expands the selected node and creates children node.
    Returns new node and updated tree with the newly created node
        Arguments:
            selected_node: selected node from selection()
            tree: contains all the nodes
        Returns:
            Tuple[Node, Tree]: returns new node and updated tree
    """

    selected_node.add_node()
    new_node = selected_node.children[len(selected_node.children) - 1]
    tree.add_to_nodes(new_node)
    new_node.player = other_player(selected_node.player)
    return new_node, tree


def simulation(newly_created_node: Node, board: np.ndarray) -> int:
    """
    Called until the game is finished (win, lost or draw)
    and return the result. When there is no valid move for the node return that the board is full.
    Arguments:
        newly_created_node: new node
        board: ndarray representation of the board
    Return:
        int: 1 for Win, -1 for Lost, 0 for Draw and  -2 for Full when the board is full
        and the newly_created_node can't make a move
    """

    opponent = other_player(newly_created_node.player)

    move = valid_move(board, opponent, newly_created_node)
    if move is None:
        newly_created_node.explored = True  # this node cant be explored further
        return FULL

    # set nodes value and move
    newly_created_node.move = move
    board_before = board.copy()
    newly_created_node.board_state = apply_player_action(board, move, opponent, False)
    newly_created_node.value = evaluate(newly_created_node, opponent, board_before)

    # start with simulation
    board_copy = board.copy()
    on_turn = newly_created_node.player

    while check_end_state(board_copy, opponent, None) == GameState.STILL_PLAYING \
            and check_end_state(board_copy, newly_created_node.player, None) == GameState.STILL_PLAYING:
        move = valid_move(board_copy, on_turn, None)
        if move is None:
            return DRAW  # no possible moves

        # do moves
        if on_turn == newly_created_node.player:
            apply_player_action(board_copy, move, newly_created_node.player)
            on_turn = opponent
        else:
            apply_player_action(board_copy, move, opponent)
            on_turn = newly_created_node.player

        # game is won
        if check_end_state(board_copy, opponent) == GameState.IS_WIN:
            return WIN
        if check_end_state(board_copy, newly_created_node.player) == GameState.IS_WIN:
            return LOST

    return DRAW


def evaluate(node: Node, opponent: BoardPiece, board_before: np.ndarray) -> int:
    """
    Return evaluation of the board after nodes move
    Arguments:
        node: selected node
        opponent: actually current player
        board: ndarray representation of the board
    Return
        int: nodes value after the move
    """
    pos_opp, m1 = get_position_mask_bitmap(opponent, node.board_state)  # position after the move
    pos_player, m2 = get_position_mask_bitmap(node.player, board_before)  # other player position before the move
    pos_player_block, m3 = get_position_mask_bitmap(node.player, node.board_state)  # position after the move

    if connected_four(pos_opp):
        return 200000
    # check after the move if we blocked the other player
    elif connected_three(pos_player, m2) and not connected_three(pos_player_block, m3):
        return 100000
    else:
        return evaluate_board(pos_opp, m1)


def evaluate_board(position: int, mask: int) -> int:
    """
    Evaluate board according to bit position
    Arguments:
         position: bit representation of board and where the player has piece
         mask: bit representation of board with all pieces
    Return:
        int: value of the board
    """
    num = number_of_connected(position, mask)
    if num == 3:
        return 10000
    elif num == 2:
        return 100
    else:
        return 1


def backpropagation(newly_created_node: Node, outcome: int, tree: Tree) -> Tuple[Node, Tree]:
    """
    Update nodes scores one by one by going up the tree until the root
    Returns the root and tree with the updated scores
    Arguments:
        newly_created_node: selected node
        outcome: outcome of simulation()
        tree: contains all the nodes
    Return:
        Tuple[Node, Tree]: updated node and tree values
    """

    newly_created_node.simulations += 1  # new node simulations

    if outcome == LOST:
        newly_created_node.wins += 1
        node = newly_created_node.parent
        while node.parent is not None:
            if newly_created_node.player != node.player:
                node.wins += 1
            node.simulations += 1
            node = node.parent
        node.simulations += 1
        if newly_created_node.player != node.player:
            node.wins += 1

    elif outcome == WIN:
        node = newly_created_node.parent
        while node.parent is not None:
            if newly_created_node.player == node.player:
                node.wins += 1
            node.simulations += 1
            node = node.parent
        node.simulations += 1
        if newly_created_node.player == node.player:
            node.wins += 1
    else:
        node = newly_created_node.parent
        while node.parent is not None:
            node.simulations += 1
            node = node.parent
        node.simulations += 1

    return node, tree


def valid_move(board: np.ndarray, player: BoardPiece, node: None) -> int:
    """
    Return a valid random move in the board
    Arguments:
        board: ndarray representation of the board
        node: current node
    """

    # list with invalid moves
    invalid_moves = []
    if node and len(node.parent.children) > 1:
        invalid_moves = [item.move for item in node.parent.children]

    actionList = [col for col in range(7) if np.count_nonzero(board[:, col] == 0) > 0 and col not in invalid_moves]

    return random.choice(actionList) if len(actionList) > 0 else None


def other_player(player: BoardPiece) -> BoardPiece:
    """
    Function returning the opponent of current player
    Arguments:
        player: current player
    Return:
        BoardPiece: opponent player
    """
    if player == BoardPiece(1):
        return BoardPiece(2)
    else:
        return BoardPiece(1)


def get_position_mask_bitmap(player: BoardPiece, board: np.ndarray) -> Tuple[int, int]:
    """
    Return bit representation of the board
    Arguments:
        player: player for whom we will return position
        board: ndarray representation of the board to be converted in bitboard
    Return:
        Tuple[int, int]: return position and mask
        position : bit representation of the board with players pieces
        mask: bit representation of the board with all pieces
    """
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


def number_of_connected(position: int, mask: int) -> int:
    """
    Returns number of connected pieces for player by given position
    Arguments:
        position: bit representation of the board with players pieces
        mask: bit representation of board with all pieces
    Return:
        int: number of connected pieces
    """
    if connected_four(position):
        return 4
    elif connected_three(position, mask):
        return 3
    elif connected_two(position, mask):
        return 2
    else:
        return 1


def connected_four(position) -> bool:
    """
    Return True, when the player has connected 4
    Arguments:
        position: bit representation of the board with players pieces
    Return:
        bool: True for 4 connected
    """
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


def connected_three(position: int, mask: int) -> bool:
    """
    Return True, when the player has connected 3 and free space after/before them
    Arguments:
        position: bit representation of the board with players pieces
        mask: bit representation of board with all pieces
    Return:
        bool: True for 3 connected
    """
    opp_pos = mask ^ position

    # Diagonal /
    m = position & (position >> 8)
    if m & (m >> 8):
        bits = "{0: 049b}".format(m & (m >> 8))
        foo = [len(bits) - i - 1 for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if ((opp_pos >> int(foo[j] + 24)) & 1) != 1 and foo[j] + 24 < 49:
                return True
            elif foo[j] >= 8 and ((opp_pos >> int(foo[j] - 8)) & 1) != 1:
                return True

    # Diagonal \
    m = position & (position >> 6)
    if m & (m >> 6):
        bits = "{0: 049b}".format(m & (m >> 6))
        foo = [len(bits) - i - 1 + 6 for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if ((opp_pos >> int(foo[j] + 12)) & 1) != 1 and foo[j] + 12 < 49:
                return True
            elif foo[j] >= 12 and ((opp_pos >> int(foo[j] - 12)) & 1) != 1:
                return True

    # Horizontal
    m = position & (position >> 7)
    b = m & (m >> 7)
    if b:
        bits = "{0: 049b}".format(b)
        foo = [len(bits) - i - 1 + 7 for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if ((opp_pos >> int(foo[j] + 14)) & 1) != 1 and foo[j] + 14 < 49:
                return True
            elif foo[j] >= 14 and ((opp_pos >> int(foo[j] - 14)) & 1) != 1:
                return True

    # Vertical
    m = position & (position >> 1)
    t = m & (m >> 1)
    if t:
        bits = "{0: 049b}".format(t)
        foo = [len(bits) - i for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if foo[j] >= 2 and ((opp_pos >> int(foo[j] - 2)) & 1) != 1:
                return True

    # Nothing found
    return False


def connected_two(position: int, mask: int) -> bool:
    """
    Return True, when the player has connected 2
    Arguments:
        position: bit representation of the board with players pieces
        mask: bit representation of board with all pieces
    Return:
        Tuple[bool,bool] = first value is true when player has 2 connected and 2 spaces after/before them,
                            second value is true when player has 2 connected, free space and one more piece (X X _ X)
    """

    opp_pos = mask ^ position
    # Diagonal /
    tmp = position & (position >> 8)
    if tmp:
        bits = "{0: 049b}".format(tmp)
        foo = [len(bits) - i - 1 for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if ((opp_pos >> int(foo[j] + 24)) & 1) != 1 and ((opp_pos >> int(foo[j] + 32)) & 1) != 1 \
                    and foo[j] + 24 < 49 and foo[j] + 32 < 49:
                return True
            elif foo[j] >= 16 and ((opp_pos >> int(foo[j] - 8)) & 1) != 1 and ((opp_pos >> int(foo[j] - 16)) & 1) != 1:
                return True
            elif ((opp_pos >> int(foo[j] + 16)) & 1) != 1 and foo[j] + 16 < 49 and foo[j] >= 16 \
                    and ((opp_pos >> int(foo[j] - 16)) & 1) != 1:
                return True

    # Diagonal \
    tmp = position & (position >> 6)
    if tmp:
        bits = "{0: 049b}".format(tmp)
        foo = [len(bits) - i - 1 + 6 for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if ((opp_pos >> int(foo[j] + 6)) & 1) != 1 and ((opp_pos >> int(foo[j] + 12)) & 1) != 1 \
                    and foo[j] + 6 < 49 and foo[j] + 12 < 49:
                return True
            elif foo[j] >= 18 and ((opp_pos >> int(foo[j] - 12)) & 1) != 1 and ((opp_pos >> int(foo[j] - 18)) & 1) != 1:
                return True

    # Horizontal
    tmp = position & (position >> 7)
    if tmp:
        bits = "{0: 049b}".format(tmp)
        foo = [len(bits) - i - 1 + 7 for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if ((opp_pos >> int(foo[j] + 7)) & 1) != 1 and foo[j] + 7 <= 49 and \
                    ((opp_pos >> int(foo[j] + 14)) & 1) != 1 and foo[j] + 14 < 49:
                return True
            elif foo[j] >= 21 and ((opp_pos >> int(foo[j] - 14)) & 1) != 1 and ((opp_pos >> int(foo[j] - 21)) & 1) != 1:
                return True
            elif foo[j] >= 14 and ((opp_pos >> int(foo[j] - 14)) & 1) != 1 and ((opp_pos >> int(foo[j] + 7)) & 1) != 1 \
                    and foo[j] + 7 < 49:
                return True

    # Vertical
    tmp = position & (position >> 1)
    if tmp:
        bits = "{0: 049b}".format(tmp)
        foo = [len(bits) - i for i in range(0, len(bits)) if bits[i] == '1']
        for j in range(len(foo)):
            if foo[j] >= 2 and ((opp_pos >> int(foo[j] - 2)) & 1) != 1:
                return True

    # Nothing found
    return False
