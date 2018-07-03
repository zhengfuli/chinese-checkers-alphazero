#implement basic Monte Carlo Tree Search algorithm

import numpy as np
from game import Game

# tree node of the Monte Carlo Tree
class TreeNode(object):
    def __init__(self, parent, prob):
        # parent node
        self.parent = parent

        # map action to child node
        self.children = {}

        # visit count
        self.N = 0

        # average action value
        self.Q = 0.0

        # overall action value
        self.W = 0.0

        # prior probability, the chance of the action that leads to this node is chosen
        self.P = prob

        # upper confidence bounds(UCB) for Trees(UCT)
        # plus bonus u to encourage exploration during searching
        self.u = 0.0

    def isLeafNode(self):
        return self.children == {}

    def isRoot(self):
        return not self.parent

class MCTS(object):
    def __init__(self, c_puct=5, sim_count, nn):
        # hyper-parameter for policy upper confidence bounds of game trees algorithm
        # larger value means more exploitation while smaller value means more exploration
        self.c_puct = c_puct

        # number of MCTS simulations
        self.sim_count = sim_count

        # policy&value network
        self.nn = nn

        # each MCTS self-play starts from an empty node
        self.root = TreeNode(None, 1.0)


    # select an action given a non-leaf node during MCTS simulation
    # a = argmax(a)(Q(s,a) + u(s,a))
    def select(self, node):
        """
        :param node: MCTS TreeNode
        :return: a tuple of action and resulting TreeNode
        """
        res = ()
        max_Q_plus_u = 0.0

        # calculate Q + U and return the action that maximizes the overall action value
        for child in node.children.items():
            node = child[1]
            u = self.c_puct * node.P * np.sqrt(node.parent.N / (1 + node.N))

            if node.Q + node.u > max_Q_plus_u:
                max_Q_plus_u = node.Q + node.u
                res = child

        return res


    # expand a leaf node given all possible actions and their probabilities
    def expand(self, node, action_probs):
        for action, prob in action_probs:
            if action not in node.children:
                child = TreeNode(node, prob)
                node.children[action] = child


    # recursively update node parameters using leaf node value in the back-up way
    # this is in the simulation process rather than self-play and a leaf node refers
    # to game ending or a node that has not been expanded in previous simulation iterations
    def backup(self, node, value):
        if node:
            # being visited once more
            node.N += 1

            # overrall value increases
            node.W += value

            # update average action value, this is the AlphaGo version, APV-MCTS
            # for alphazero, it's like node.Q += 1.0*(value-node.Q)/node.N
            node.Q = node.W / node.N

            # update bouns u
            node.u = self.c_puct * node.P * np.sqrt(node.parent.N / (1 + node.N))

            # recursively update
            # the tree is made of nodes that represent both you and your rival's actions
            # and thus value should be reversed alternatively during recursion
            self.backup(node.parent, -value)