import datetime
import os

import chess
import numpy
import torch
from stockfish import Stockfish

from games.abstract_game import AbstractGame
from move_mapper import uci_moves, uci_to_index


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough
        # memory. None will use every GPUs available

        ### Game
        self.observation_shape = (2, 8, 8)  # Dimensions of the game observation, must be 3D (channel, height, width).
        # For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(
            range(len(uci_moves)))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = 'self'  # Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        # It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = torch.cuda.is_available()
        self.max_moves = 512  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3  # https://medium.com/applied-data-science/how-to-build-your-own-deepmind-muzero-in-python-part-2-3-f99dad7a7ad
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range
        # of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter)
        # / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 16  # Number of channels in reward head
        self.reduced_channels_value = 16  # Number of channels in value head
        self.reduced_channels_policy = 16  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "../results", os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the
        # model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 10000000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 512  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 2e-1  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 100000

        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 4  # Number of game moves to keep for every batch element
        self.td_steps = 20  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.env = Chess()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
    
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.get_legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        print(self.env.board)
        while True:
            try:
                legal_moves = self.env.get_legal_moves()

                print(str(legal_moves))

                move_chosen = input(
                    f"Enter move for {'White' if self.to_play() == PLAYER_WHITE else 'Black'}: "
                )
                if move_chosen in legal_moves:
                    print(move_chosen)
                    return uci_to_index[move_chosen]
            except:
                pass
            print("Wrong input, try again")

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        
        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """

        return uci_moves[action_number]


PLAYER_WHITE = 0
PLAYER_BLACK = 1


class Chess:
    def __init__(self):
        self.board = chess.Board()
        self.result = None
        self.player = PLAYER_WHITE
        self.moves = 0
        self.stock_fish = Stockfish('stockfish/stockfish_13_linux_x64_bmi2',
                                    parameters={"Threads": 2, "Minimum Thinking Time": 200})
        self.stock_fish.set_elo_rating(3000)

    def to_play(self):
        self.stock_fish.set_fen_position(self.board.fen())
        return self.player

    def reset(self):
        self.board = chess.Board()
        self.result = None
        self.player = PLAYER_WHITE
        self.moves = 0

        return self.get_observation()

    def step(self, action: int):
        self.board.push_uci(uci_moves[action])
        self.moves += 1
        reward = 0
        best_move = self.stock_fish.get_best_move()
        best_action = uci_to_index[best_move]

        if action == best_action:
            reward = 1

        # print(f'Moves: {self.moves} - Player: {self.player} - Move: {uci_moves[action]} - Reward: {reward}')

        if self._is_game_over():
            print(f'Game ended - {self.result} - total moves: {self.moves}')
            return self.get_observation(), reward, True

        self._set_next_player()

        return self.get_observation(), 0, False

    def get_legal_actions(self):
        legal_moves = []

        for legal_move in self.board.legal_moves:
            action_num = uci_to_index[legal_move.uci()]
            legal_moves.append(action_num)

        return legal_moves

    def get_legal_moves(self):
        legal_moves = []

        for legal_move in self.board.legal_moves:
            legal_moves.append(legal_move.uci())

        return legal_moves

    def get_observation(self):
        int_board = self._convert_to_int_board()
        to_move = [[self.to_play()] * 8] * 8

        return numpy.array([int_board, to_move])

    def expert_action(self):
        return uci_to_index[self.stock_fish.get_best_move()]

    def render(self):
        print(self.board)

    def _is_game_over(self):
        if self.board.can_claim_draw() or self.board.can_claim_fifty_moves() or \
                self.board.can_claim_threefold_repetition():
            self.result = "draw"
            return True
        elif self.board.legal_moves.count() == 0:
            if self.player == PLAYER_WHITE:
                self.result = 'Winner: Black'
            else:
                self.result = 'Winner: White'
            return True

        return False

    def _set_next_player(self):
        if self.player == PLAYER_WHITE:
            self.player = PLAYER_BLACK
        else:
            self.player = PLAYER_WHITE

    def _convert_to_int_board(self):
        int_board = []
        rows = self.board.__str__().split('\n')

        for row in rows:
            int_row = []
            for field in row.split(' '):
                if field == 'k':
                    int_row.append(6)
                elif field == 'K':
                    int_row.append(-6)
                if field == 'r':
                    int_row.append(5)
                elif field == 'R':
                    int_row.append(-5)
                if field == 'n':
                    int_row.append(4)
                elif field == 'N':
                    int_row.append(-4)
                if field == 'b':
                    int_row.append(3)
                elif field == 'B':
                    int_row.append(-3)
                if field == 'p':
                    int_row.append(2)
                elif field == 'P':
                    int_row.append(-2)
                if field == 'q':
                    int_row.append(1)
                elif field == 'Q':
                    int_row.append(-1)
                elif field == '.':
                    int_row.append(0)

            int_board.append(int_row)
        return int_board
