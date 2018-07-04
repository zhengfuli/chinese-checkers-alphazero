#implement basic Monte Carlo Tree Search algorithm

import numpy as np
import settings

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
    def __init__(self, nn):

        # policy&value network
        self.nn = nn

        # each MCTS simulation starts from am empty root, the only difference is what state in the self-play
        # is this root corresponding to in this simulation
        prob = (1.0 - settings.noise_eps) * 1.0 + settings.noise_eps * np.random.dirichlet([settings.dirichlet_alpha * 1.0])
        self.root = TreeNode(None, prob)


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
            u = settings.c_puct * node.P * np.sqrt(node.parent.N / (1 + node.N))

            if node.Q + u > max_Q_plus_u:
                max_Q_plus_u = node.Q + u
                res = child

        return res


    # expand a leaf node given all possible actions and their probabilities
    def expand(self, node, action_probs):
        """
        :param node: leaf node that has no children
        :param action_probs: valid actions and probabilities predicted by NN for this node (represents a state)
        :return: add children in-place
        """
        for action, prob in action_probs:
            if action not in node.children:
                child = TreeNode(node, prob)
                node.children[action] = child


    # recursively update node parameters using leaf node value in the back-up way
    # this is in the simulation process rather than self-play and a leaf node refers
    # to game ending or a node that has not been expanded in previous simulation iterations
    def backup(self, node, value):
        """
        :param node: tree nodes that are in the path to a leaf node
        :param value: leaf node value or the opposite value
        :return: update in-place
        """
        if node:
            # being visited once more
            node.N += 1

            # overall value increases
            node.W += value

            # update average action value, this is the AlphaGo version, APV-MCTS
            # for AlphaZero, it's like node.Q += 1.0*(value-node.Q)/node.N
            node.Q = node.W / node.N

            # update bonus u
            node.u = settings.c_puct * node.P * np.sqrt(node.parent.N / (1 + node.N))

            # recursively update
            # the tree is made of nodes that represent both you and your rival's actions
            # and thus value should be alternatively opposite during recursion
            self.backup(node.parent, -value)



    # select, expand to search for a leaf node and then update
    # this is one simulation process, state is the board state in the self-play when this simulation starts
    def search(self, state):
        node = self.root
        while not node.isLeafNode():
            # select action when this node is expanded
            action, node = self.select(node)
            # update state using this action
            state.execute_move(action)

        # assign the value when game is ended
        if state.isEnded():
            winner = state.getWinner()
            # winner = 0 for tie, 1 for player 1 and 2 for player 2
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if winner == state.getCurretnPlayer() else -1.0
        else:
            # evaluate the leaf node using neural network while game is not ended
            action_probs, value = self.nn(node)
            # expand this leaf node
            self.expand(node, action_probs)

        # update nodes' weights in the path to this leaf node
        self.backup()