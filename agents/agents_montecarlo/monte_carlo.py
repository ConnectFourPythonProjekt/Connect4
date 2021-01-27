from agents.agents_montecarlo.tree_initialization import Tree, Node


def selection(tree: Tree) -> Node:
    """
    Returns the node that has the highest possibility of winning (UCB1)
    Called by expansion
    """
    return None


def expansion(tree: Tree, selected_node: Node) -> (Tree, Node):
    """
    The function expands the selected node and creates many children nodes.
    Returns the tree with the newly created node
    and the newly created node
    Called by simulation
    """
    return None, None


def simulation(tree: Tree, newly_created_node: Node) -> Tree:
    """
    Recursivly called until the game is finished and a winner emerges
    Returns the tree with the new nodes
    and their positive or negative scores
    """
    return None


def backpropagation(tree: Tree) -> Tree:
    """
    Update the parent scores one by one by going up the tree
    Returns the tree with the updated scores
    """
    return None
