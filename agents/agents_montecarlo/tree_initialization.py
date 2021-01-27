import numpy as np


class Tree():
    def __init__(self, root):
        self.root = root
        self.children = []
        self.Nodes = []

    def add_node(self, new_node):
        self.children.append(new_node)

    def get_all_nodes(self):
        self.Nodes.append(self.root)
        for child in self.children:
            self.Nodes.append(child.score)
        for child in self.children:
            if child.get_child_nodes(self.Nodes) is not None:
                child.get_child_nodes(self.Nodes)


class Node():
    def __init__(self, score, board_state,move, parent):
        self.score = score
        self.children = []
        self.board_state = board_state
        self.move = move
        self.parent = parent
        self.simulations = 0

    def add_node(self, new_node):
        self.children.append(new_node)

    def get_child_nodes(self, Tree):
        for child in self.children:
            if child.children:
                child.get_child_nodes(Tree)
                Tree.append(child.score)
            else:
                Tree.append(child.score)
