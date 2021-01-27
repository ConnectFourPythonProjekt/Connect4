import numpy as np


class Node():
    def __init__(self, score, board_state, move, parent):
        self.score = score
        self.children = []
        self.board_state = board_state
        self.move = move
        self.parent = parent
        self.simulations = 0
        self.wins = 0


    def add_node(self, new_node):
        self.children.append(new_node)


def upper_confidence_bound(current_node: Node) -> float:
    w = current_node.wins
    s = current_node.simulations
    sp = current_node.parent.simulations
    c = np.sqrt(2)

    return (w / s) + c * (np.sqrt((np.log(sp)) / s))


def selection(root: Node) -> Node:
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
    return best_node


def expansion(selected_node: Node) -> list:
    """
    The function expands the selected node and creates many children nodes.
    Returns the tree with the newly created node
    and the newly created node
    Called by simulation
    """
    return None, None


def simulation(newly_created_node: Node) -> Node:
    """
    Recursivly called until the game is finished and a winner emerges
    Returns the tree with the new nodes
    and their positive or negative scores
    """
    return None


def backpropagation() -> Node:
    """
    Update the parent scores one by one by going up the tree
    Returns the tree with the updated scores
    """
    return None
