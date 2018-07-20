#implement basic Monte Carlo Tree Search algorithm
import time
import numpy as np
import settings
import copy

# tree node of the Monte Carlo Tree
class TreeNode(object):
    def __init__(self, parent, prob):
        # parent node
        self.parent = parent

        # map a move to a resulting child node
        self.children = {}

        # visit count
        self.N = 0

        # average move value
        self.Q = 0.0

        # prior probability, the chance that the move which leads to this node is chosen
        self.P = prob

        # upper confidence bounds(UCB) for Trees(UCT) algorithm
        # average plus bonus u to encourage exploration during searching
        self.u = 0.0

    def _isLeafNode(self):
        return self.children == {}

    def _isRoot(self):
        return not self.parent


class MCTS(object):
    def __init__(self, nn):
        np.random.seed(int(time.time()))

        # policy&value network
        self.nn = nn

        # each MCTS simulation starts from am empty root, the only difference is what state of board
        # in the self-play is this root corresponding to in this simulation
        prob = (1.0 - settings.noise_eps) * 1.0 + settings.noise_eps * np.random.dirichlet([settings.dirichlet_alpha * 1.0])
        self.root = TreeNode(None, prob)

    # select a move given a non-leaf node during MCTS simulation
    # a = argmax(a)(Q(s,a) + u(s,a))
    def _select(self, node):
        """
        :param node: MCTS TreeNode
        :return: a tuple of move and resulting TreeNode
        """
        res = ()
        max_Q_plus_u = 0.0

        # calculate Q + U and return the move that maximizes the overall action(move) value
        for child in node.children.items():
            node = child[1]
            u = settings.c_puct * node.P * np.sqrt(node.parent.N / (1 + node.N))

            if node.Q + u > max_Q_plus_u:
                max_Q_plus_u = node.Q + u
                res = child

        return res

    # expand a leaf node given all possible moves and their probabilities
    def _expand(self, node, valid_moves, move_probs):
        """
        :param node: leaf node that has no children
        :param valid_moves: valid moves at this state
        :param move_probs: all move probabilities predicted by NN for this node (represents a state)
        :return: add children in-place
        """
        # calculate total probabilities of valid moves for normalizing
        valid_move_probs = 0.0
        # only expand valid moves and then normalize probabilities
        for move in valid_moves:
            assert(move in move_probs)
            prob = move_probs[move]
            valid_move_probs += prob
            child = TreeNode(node, prob)
            node.children[move] = child

        # if there is no valid move now, the node's children should be empty
        if valid_move_probs == 0.0:
            assert(node.children  == {})
        else:
            # normalize probabilities
            for move, child_node in node.children.items():
                child_node.P /= valid_move_probs

    # recursively update node parameters using leaf node value in the back-up way
    # this is in the simulation process rather than self-play and a leaf node refers
    # to game ending or a node that has not been expanded in previous simulation iterations
    def _backup(self, node, value):
        """
        :param node: tree nodes that are in the path to a leaf node
        :param value: leaf node value or the opposite value
        :return: update in-place
        """
        cur_node = node
        if cur_node:
            # being visited once more
            cur_node.N += 1

            # update average action value
            cur_node.Q += 1.0 * (value - cur_node.Q) / cur_node.N

            # update bonus u
            cur_node.u = settings.c_puct * cur_node.P * np.sqrt(cur_node.parent.N / (1 + cur_node.N))

            # recursively update
            # the tree is made of nodes that represent both you and your rival's actions
            # and thus value should be alternatively opposite during recursion
            self._backup(cur_node.parent, -value)

    # select, expand to search for a leaf node and then update
    # this is one complete simulation process
    def _search(self, state):
        """
        :param state: board state in the self-play process when this simulation starts
        :return: modify the Monte Carlo tree in-place
        """
        # do not modify the argument state as you are just conducting simulations
        # while this state cannot be changed before all simulations are done
        sim_state = copy.deepcopy(state)

        node = self.root
        while not node.isLeafNode():
            # select a move when this node is already expanded
            move, node = self._select(node)
            # update state executing this move
            sim_state.executeMove(move)

        # assign the value when game is ended
        if sim_state.isEnded():
            winner = sim_state.getWinner()
            # winner = 0 for a tie, 1 for player 1 and 2 for player 2
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if winner == sim_state.getCurretnPlayer() else -1.0
        else:
            # get valid actions/moves at this state when the game is not ended yet
            valid_moves = sim_state.getValidMoves()

            # evaluate the leaf node using neural network while game is not ended
            # the move probabilities will cover both valid and invalid moves
            # but expand function will handle
            move_probs, value = self.nn(node)

            # expand this leaf node
            self._expand(node, valid_moves, move_probs)

        # update nodes' weights in the path to this leaf node
        # starting from the leaf node's parent
        self._backup(node, -value)

    def _simulate(self, state):
        """
        :param state: current state board
        :return: execute assigned number of MCTS simulations and return the policy
        """
        for n in range(settings.sim_count):
            self._search(state)

        # calculate policy/probability distribution over the state after simulations using softmax
        moves = list(self.root.children.keys())
        visit_count = list(self.root.children.values())

        # prob(xi) = exp(ni) / sum(exp(ni))
        def softmax(n):
            probs = np.exp(n - np.max(n))
            probs /= np.sum(probs)
            return probs

        # pi(a|s) = N(s,a)**(1/temp) / sum(i)(N(s,ai)**(1/temp))
        # when visit times is 1 log() can be zero thus plus a bias
        probs = softmax(1.0 / settings.temperature * np.log(np.array(visit_count)) + 1e-10)

        return moves, probs

    # after executing a certain move in the self-play, the MCST should be updated
    def _update_tree(self, move):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

# handle actions between two players when using MCTS class
class MCTSPlayer(object):
    def __init__(self, selfplay, nn):
        self.selfplay = selfplay
        self.nn = nn
        self.MCTS = MCTS(self.nn)

    def _set_player_id(self, id):
        self.player_id = id

    def _act(self, state):
        moves, probs = self.MCTS._simulate(state)

        if self.selfplay:
            # add dirichlet noise for better exploration
            move = np.random.choice(moves, p=(1.0-settings.noise_eps)*probs + \
                   settings.noise_eps*np.random.dirichlet(settings.dirichlet_alpha*np.ones(len(probs))))
        else:
            move = np.random.choice(moves, p=probs)

        self.MCTS._update_tree(move)
        return move
