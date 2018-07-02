import numpy as np

# tree node of the Monte Carlo Tree
class TreeNode(object):
    def __init__(self, parent, prob):
        # parent node
        self.parent = parent

        # map action to child node
        self.children = {}

        # visit count
        self.N = 0

        # action value
        self.Q = 0.0

        # prior probability
        self.P = prob

        # upper confidence bounds(UCB) for Trees(UCT)
        # plus bonus u to encourage exploration during searching
        self.u = 0.0

class MCTS(object):
    def __init__(self, c_puct, sim_count, nn):
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
            node.u = self.c_puct * node.P * np.sqrt(node.parent.N / (1 + node.N))

            if node.Q + node.u > max_Q_plus_u:
                max_Q_plus_u = node.Q + node.u
                res = child

        return res

