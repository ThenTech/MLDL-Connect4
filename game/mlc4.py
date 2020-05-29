from colours import Colours, style

# seed = 1232611  # Student number Cedric
seed = 0

import random
if seed: random.seed(seed)

import numpy as np
if seed: np.random.seed(seed)

from connectfour import Game, starting_player

import os, sys

import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
if seed: tf.random.set_seed(seed)

import keras
from keras.models import load_model, save_model
from keras.layers import Dense
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Nadam

import matplotlib.pyplot as plt


class MLC4:
    def __init__(self, board=None, print_friendly=False):
        super().__init__()
        self.game = Game(board=board)  # Cannot do inheritance due to deepcopy in Game...
        self.states = []

        self.model = None
        self.board_nodes = self.game.width * self.game.height
        self.input_shape  = (self.board_nodes + 1,)

        # If True, use another character for the other player so they are distinct
        # when printing without colours, e.g. ■ ▀ • ⦿
        self.print_friendly = print_friendly
        self.__players = {
            -1: style("■", Colours.FG.YELLOW),
             0: " ",
             1: style("•" if self.print_friendly else "■", Colours.FG.RED)
        }

    def __str__(self):
        top_row = "┌" + ("─" * 3 + "┬") * (self.game.width-1) + "─" * 3 + "┐"
        sep_row = "╞" + ("═" * 3 + "╪") * (self.game.width-1) + "═" * 3 + "╡"
        bot_row = "└" + ("─" * 3 + "┴") * (self.game.width-1) + "─" * 3 + "┘"

        return top_row \
             + "\n" + "\n".join(f"│ {f' │ '.join(map(self.print_player, li))} │" for li in reversed(self.game.board)) \
             + "\n" + sep_row \
             + "\n│ " + f' │ '.join(map(str, range(self.game.width))) + " │" \
             + "\n" + bot_row

    def _add_state(self, game, player, move):
        if player is not None and move is not None:
            self.states.append((player, move))
            print(self)

    def print_states(self, game_id=0, fp=None):
        for idx, move in enumerate(self.states):
            print(game_id, idx, move[0], move[1], self.game.status, sep=",", file=fp or sys.stdout)

    def print_player(self, p):
        return self.__players.get(p, self.__players[0])

    def print_player_string(self, p):
        if p == 0:
            return "Player DRAW"
        return f"Player {max(0, p) + 1} " + self.print_player(p)

    def reset(self):
        self.game = Game()
        self.states = []

    def play_original_vs_random(self, starting=None, legal_only=True, n=100):
        # Dummy call original play function
        return self.game.random_play(starting, legal_only, self._add_state)

    def play_original_vs_smart(self, starting=None, legal_only=True, n=100):
        # Dummy call original play function
        return self.game.smart_play(starting, legal_only, n, self._add_state)

    def _start_game(self, player, other_strat_descr="plays randomly"):
        print("-" * 80 + f"\n\nStarting game with {self.print_player_string(player)}. " \
            + f"AI player is {self.print_player_string(-1)}." \
            + f" {self.print_player(1)} {other_strat_descr}.")
        self._add_state(self.game, None, None)

    def play_vs_random(self, starting=None, legal_only=True, check_early_win=True, prevent_other_win=True):
        # Against random player
        player = starting if starting is not None else starting_player()
        self._start_game(player, other_strat_descr="plays randomly")

        while self.game.status is None:
            if player < 0 and self.model:
                move = self.predict(check_early_win=check_early_win,
                                    prevent_other_win=prevent_other_win)
            else:
                move = self.game.random_action(legal_only=legal_only)

            print(f"{self.print_player_string(player)} adds to column {move}...")
            self.game.play_move(player, move)
            self._add_state(self.game, player, move)
            player = player * -1

        print(f"{self.print_player_string(self.game.status)} wins!")
        return self.game.status

    def play_vs_smart(self, starting=None, legal_only=True, n=100, check_early_win=True, prevent_other_win=True):
        # Against smart player
        player = starting if starting is not None else starting_player()
        self._start_game(player, other_strat_descr="plays smart")

        while self.game.status is None:
            if player < 0 and self.model:
                move = self.predict(check_early_win=check_early_win,
                                    prevent_other_win=prevent_other_win)
            else:
                move, p = self.game.smart_action(player, legal_only=legal_only, n=n)
                if not self.game.is_legal_move(move):
                    print(style("Illegal move smart player! ", Colours.FG.BRIGHT_RED), player, move)

            print(f"{self.print_player_string(player)} adds to column {move}...")
            self.game.play_move(player, move)
            self._add_state(self.game, player, move)
            player = player * -1

        print(f"{self.print_player_string(self.game.status)} wins!")
        return self.game.status

    def play_vs_ai(self, starting=None, legal_only=True, check_early_win=True, prevent_other_win=True, random_move_chance=0.0):
        # Against own model
        player = starting if starting is not None else starting_player()
        self._start_game(player, other_strat_descr="plays also as AI")

        while self.game.status is None:
            if player > 0 and random_move_chance > 0.0 and random.random() <= random_move_chance:
                move = self.game.random_action(legal_only=legal_only)
            else:
                move = self.predict(ai_player=player,
                                    check_early_win=check_early_win,
                                    prevent_other_win=prevent_other_win)

            if not self.game.is_legal_move(move):
                print(style("Illegal move from player! ", Colours.FG.BRIGHT_RED), player, move)

            print(f"{self.print_player_string(player)} adds to column {move}...")
            self.game.play_move(player, move)
            self._add_state(self.game, player, move)
            player = player * -1

        print(f"{self.print_player_string(self.game.status)} wins!")
        return self.game.status

    ###########################################################################

    def has_model(self):
        return self.model is not None

    def load_existing_model(self, name, basepath="../data/models/"):
        try:
            self.model = load_model(f"{basepath}{name}", compile=True)
            self.model.predict(np.zeros((1, *self.input_shape)))  # Init predictor
        except Exception as e:
            self.model = None
            print(style(f"Could not load model!\n{e}", Colours.FG.RED))
        else:
            self.model.summary()

    def build_network(self, name="", learning_rate=0.001):
        """
        Input : self.width * self.height board (42 squares) + player

            https://keras.io/api/layers/activations/#relu-function
            https://keras.io/api/layers/activations/#softmax-function
            https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class
            https://keras.io/api/optimizers/Nadam/
            https://keras.io/api/metrics/accuracy_metrics/#accuracy-class

        Output: 2 => [[player_0_prob, player_1_prob]]
        """
        print(f"Building model{(' ' + name) if name else ''}...")

        self.model = keras.Sequential(name=name or None)

        # Input layer
        self.model.add(Dense(self.input_shape[0], input_dim=self.input_shape[0]))

        # One or more large layers
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dense(64, activation='relu'))

        # Smaller ending layer
        self.model.add(Dense(self.board_nodes, activation='relu'))
        # self.model.add(Dense(self.game.width, activation='relu'))

        # Output end layer
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss=SparseCategoricalCrossentropy(),
                           optimizer=Nadam(learning_rate=learning_rate),
                           metrics=["accuracy"])

        self.model.summary()

    def prepare_data(self, input_file, train_ratio=0.8):
        print("Preparing data...")

        # Read data, should contain [ (board:42, winner:1, player:1), ... ]
        data = np.load(input_file)

        # Board and player values are made positive by adding one
        data += 1

        Y = data[:, -2:-1]          # (winner), result is 0 or 2
        X = np.delete(data, -2, 1)  # Drop second to last column, result = (board state, player)

        size = int(train_ratio * X.shape[0])

        X_train, X_test, Y_train, Y_test = X[:size], X[size:], Y[:size], Y[size:]

        print("Data loaded.")
        return (X_train, Y_train), (X_test, Y_test)

    def train(self, train_data, test_data, epochs=10, batch_size=200, show_plot=False, save_plot_path=""):
        print("Training model...")

        train_x, train_y = train_data

        hist = self.model.fit(train_x, train_y,
                              validation_data=test_data,
                              shuffle=True,
                              epochs=epochs,
                              batch_size=batch_size)

        test_score, test_acc = self.model.evaluate(test_data[0], test_data[1], verbose=0)

        print(style("Final accuracy on training set : ", Colours.FG.MAGENTA) \
            + style(f"{hist.history['accuracy'][-1] * 100:.2f}%", Colours.FG.BRIGHT_MAGENTA))
        print(style("Average accuracy while training: ", Colours.FG.MAGENTA) \
            + style(f"{np.average(np.array(hist.history['val_accuracy'])) * 100:.2f}%", Colours.FG.BRIGHT_MAGENTA))
        print(style("Average accuracy on test set   : ", Colours.FG.MAGENTA) \
            + style(f"{test_acc * 100:.2f}% (score={test_score:.4f})", Colours.FG.BRIGHT_MAGENTA))

        if show_plot or save_plot_path:
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(0, epochs), [1.0] * epochs, "r--", label="Accuracy target", )
            plt.plot(np.arange(0, epochs), hist.history["loss"], "cyan", label="train_loss")
            plt.plot(np.arange(0, epochs), hist.history["val_loss"], "blue", label="val_loss")
            plt.plot(np.arange(0, epochs), hist.history["accuracy"], "yellow", label="train_acc")
            plt.plot(np.arange(0, epochs), hist.history["val_accuracy"], "orange", label="val_acc")

            # Optionally plot other metrics?
            for k, v in hist.history.items():
                if k not in ("loss", "val_loss", "accuracy", "val_accuracy"):
                    plt.plot(np.arange(0, epochs), v, label=k)

            plt.title("Training Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.xlim(0, epochs - 1)
            plt.ylim(bottom=0)
            if save_plot_path: plt.savefig(save_plot_path)
            if show_plot: plt.show()

        self.model.predict(np.zeros((1, *self.input_shape)))  # Init predictor

    def save_model(self, name="trained_1", basepath="../data/models/", save_structure=False):
        # if not name.endswith(".h5"):
        #     name += ".h5"
        print(f"Saving model to: '{basepath}{name}'...")

        if not os.path.exists(basepath):
            os.makedirs(basepath)

        if save_structure:
            model_json = self.model.to_json()
            with open(f"{basepath}{name}.json", "w") as fp:
                fp.write(model_json)

        save_model(self.model, f"{basepath}{name}")
        print("Saving complete.")

    def _predict_move_probability(self, player, check_early_win=True):
        # Predict chance of winning for each move
        # and return column with highest chance.
        max_probability = (0, 0.0)

        print(self.print_player_string(player) + ": Testing cols: |", end="")

        for move in range(self.game.width):
            if not self.game.is_legal_move(move):
                print(style(f" {0.0:.3f} |", Colours.FG.BRIGHT_RED), end="")
                continue

            test_game = Game(board=self.game.board.copy())
            test_game.play_move(player, move)

            if check_early_win and test_game.status == player:
                # Win reached
                print(" win", end="")
                max_probability = (move, 1.0)
                break

            # Get prediction for move (make board positive by adding 1)
            test_input = np.concatenate((test_game.board.flatten(), [player])).reshape((1, *self.input_shape)) + 1
            prediction = self.model.predict(test_input)[0][player + 1]  # [[player_0_prob, draw (?), player_1_prob]]

            if np.isnan(prediction):
                raise Exception("Error: prediction is NaN?")

            print(f" {prediction:.3f} |", end="")

            if prediction > max_probability[1]:
                max_probability = (move, prediction)

        print(f"  => Predicted move at col {max_probability[0]} with {max_probability[1] * 100:.2f}%")
        return max_probability

    def predict(self, ai_player=-1, check_early_win=True, prevent_other_win=True):
        # Get (move, chance) that AI wins
        ai_move = self._predict_move_probability(ai_player, check_early_win)

        if prevent_other_win:
            other_player_move = self._predict_move_probability(ai_player * -1, check_early_win)

            if other_player_move[1] > ai_move[1]:
                print(style("Trying to prevent", Colours.FG.BRIGHT_RED) \
                    + f" {self.print_player_string(ai_player * -1)} from winning...")
                return other_player_move[0]

        return ai_move[0]


class MLC4Normalised(MLC4):
    def __init__(self, board=None):
        super().__init__(board=board)
        self.input_shape = (self.board_nodes,)

    def prepare_data(self, input_file, train_ratio=0.8):
        print("Preparing data...")

        data = np.load(input_file)

        player = data[:, -1:]

        # Multiply winner and board with player to normalise player to player 1
        Y = data[:, -2:-1] * player + 1   # (winner), add 1 to make them positive
        X = data[:, :-2] * player + 1     # Drop last 2 cols, normalise and make positive, result = (board state)

        size = int(train_ratio * X.shape[0])

        X_train, X_test, Y_train, Y_test = X[:size], X[size:], Y[:size], Y[size:]

        print("Data loaded.")
        return (X_train, Y_train), (X_test, Y_test)


    def _predict_move_probability(self, player, check_early_win=True):
        # Predict chance of winning for each move
        # and return column with highest chance.
        max_probability = (0, 0.0)

        print(self.print_player_string(player) + ": Testing cols: |", end="")

        for i, move in enumerate(range(self.game.width)):
            if not self.game.is_legal_move(move):
                print(style(f" {0.0:.2f} |", Colours.FG.BRIGHT_RED), end="")
                continue

            test_game = Game(board=self.game.board.copy())
            test_game.play_move(player, move)

            if check_early_win and test_game.status == player:
                # Win reached
                print(" win", end="")
                max_probability = (i, 1.0)
                break

            # Get prediction for move
            test_input = test_game.board.flatten().reshape((1, *self.input_shape)) * player + 1
            prediction = self.model.predict(test_input)[0][player + 1]  # [[player_0_prob, draw (?), player_1_prob]]

            if np.isnan(prediction):
                raise Exception("Error: prediction is NaN?")

            print(f" {prediction:.3f} |", end="")

            if prediction > max_probability[1]:
                max_probability = (i, prediction)

        print(f"  => Predicted move at col {max_probability[0]} with {max_probability[1] * 100:.3f}%")
        return max_probability


class MLC4NormalisedConv(MLC4):
    def __init__(self, board=None):
        super().__init__(board=board)
        self.input_shape = (self.game.height, self.game.width, 1)

    def build_network(self, name="", learning_rate=0.001):
        """
        Input : self.width * self.height board (7 * 6 squares)
        Output: 2 => [[player_0_prob, player_1_prob]]
        """
        print(f"Building model{(' ' + name) if name else ''}...")

        self.model = keras.Sequential(name=name or None)

        # Input layer
        self.model.add(keras.layers.Conv2D(8, kernel_size=self.game.consecutive // 2, activation='relu',
                                           input_shape=self.input_shape,
                                           data_format="channels_last"))
        # self.model.add(keras.layers.MaxPool2D(2, strides=1))

        # One or more large layers
        # self.model.add(keras.layers.Conv2D(64, kernel_size=self.game.consecutive // 2, activation='relu'))

        self.model.add(keras.layers.Flatten())
        # self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dense(64, activation='relu'))

        # Smaller ending layer
        self.model.add(Dense(self.board_nodes, activation='relu'))

        # Output end layer
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss=SparseCategoricalCrossentropy(),
                           optimizer=Nadam(learning_rate=learning_rate),
                           metrics=["accuracy"])

        self.model.summary()


    def prepare_data(self, input_file, train_ratio=0.8):
        print("Preparing data...")

        data = np.load(input_file)

        player = data[:, -1:]

        # Multiply winner and board with player to normalise player to player 1
        Y = (data[:, -2:-1] * player + 1) / 2.0   # (winner), add 1 to make them positive
        X = (data[:, :-2] * player + 1) / 2.0     # Drop last 2 cols, normalise and make positive, result = (board state)

        # Reshape to create a matrix again
        X = X.reshape((-1, *self.input_shape))

        size = int(train_ratio * X.shape[0])

        X_train, X_test, Y_train, Y_test = X[:size], X[size:], Y[:size], Y[size:]

        print("Data loaded.")
        return (X_train, Y_train), (X_test, Y_test)


    def _predict_move_probability(self, player, check_early_win=True):
        # Predict chance of winning for each move
        # and return column with highest chance.
        max_probability = (0, 0.0)

        print(self.print_player_string(player) + ": Testing cols: |", end="")

        for i, move in enumerate(range(self.game.width)):
            if not self.game.is_legal_move(move):
                print(style(f" {0.0:.2f} |", Colours.FG.BRIGHT_RED), end="")
                continue

            test_game = Game(board=self.game.board.copy())
            test_game.play_move(player, move)

            if check_early_win and test_game.status == player:
                # Win reached
                print(" win", end="")
                max_probability = (i, 1.0)
                break

            # Get prediction for move
            test_input = (test_game.board.reshape((-1, *self.input_shape)) * player + 1) / 2.0
            prediction = self.model.predict(test_input)[0][player + 1]  # [[player_0_prob, draw (?), player_1_prob]]

            if np.isnan(prediction):
                raise Exception("Error: prediction is NaN?")

            print(f" {prediction:.3f} |", end="")

            if prediction > max_probability[1]:
                max_probability = (i, prediction)

        print(f"  => Predicted move at col {max_probability[0]} with {max_probability[1] * 100:.3f}%")
        return max_probability


class MLC4Simplified(MLC4):
    def __init__(self, board=None):
        super().__init__(board=board)

    def build_network(self, name="", learning_rate=0.001):
        """
        Input : self.width * self.height board (42 squares) + player
        Output: 2 => [[player_0_prob, player_1_prob]]
        """
        print(f"Building model{(' ' + name) if name else ''}...")

        self.model = keras.Sequential(name=name or None)

        # Input layer
        self.model.add(Dense(self.input_shape[0], input_dim=self.input_shape[0]))

        # Larger hidden layer
        self.model.add(Dense(self.board_nodes * 2, activation='relu'))

        # Output end layer
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(loss=SparseCategoricalCrossentropy(),
                           optimizer=Nadam(learning_rate=learning_rate),
                           metrics=["accuracy"])

        self.model.summary()
