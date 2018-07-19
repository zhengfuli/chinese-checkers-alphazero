import numpy as np

# hyper-parameter for policy upper confidence bounds of game trees algorithm in MCTS
# larger value means more exploitation while smaller value means more exploration
c_puct = 5

# hyper-parameter that controls the degree of exploration when employing actions decided by MCTS simulations
temperature = 1e-3

# number of MCTS simulations, 1600 in AlphaGo
sim_count = 1200

# dirichlet noise parameters when selecting in the simulation for better exploration
noise_eps = 0.25
dirichlet_alpha = 0.3

# designing virtual loss in the calculation of W and N for parallel MCTS
virtual_loss = 3