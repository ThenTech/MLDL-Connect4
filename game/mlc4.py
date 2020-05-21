from connectfour import Game, starting_player, argmax

import tensorflow as tf

import keras
from keras.layers import Dense, BatchNormalization
from keras.models import load_model, save_model

import numpy as np
import os
import sys
import contextlib
import time
from tqdm import tqdm
import colorama
colorama.init()


class HidePrintingContext:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


class MLC4:
    def __init__(self, board=None):
        super().__init__()
        self.game = Game(board=board)  # Cannot do inheritance due to deepcopy in Game...
        self.states = []

        self.model = None
        self.board_nodes = self.game.width * self.game.height

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

    def print_states(self):
        for idx, move  in enumerate(self.states):
            print(0, idx, move[0], move[1], self.game.status, sep=",")

    def print_player(self, p):
        return f"{colorama.Fore.YELLOW if p < 0 else colorama.Fore.RED}▀{colorama.Style.RESET_ALL}" \
            if p != 0 else " "

    def print_player_string(self, p):
        return f"Player {max(0, p) + 1} " + self.print_player(p)

    def reset(self):
        self.game = Game()
        self.states = []

    def play_original(self, starting=None, legal_only=True, n=100):
        # Dummy call original play function
        return self.game.smart_play(starting, legal_only, n, self._add_state)

    def _start_game(self, player):
        print("-" * 80 + f"\n\nStarting game with {self.print_player_string(player)}. " \
              + f"AI player is {self.print_player_string(-1)}." \
              + f" {self.print_player(1)} plays randomly.")
        self._add_state(self.game, None, None)

    def play_vs_random(self, starting=None, legal_only=True, check_early_win=True, prevent_other_win=True):
        # Against random player
        player = starting if starting is not None else starting_player()
        self._start_game(player)

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
        self._start_game(player)

        while self.game.status is None:
            if player < 0 and self.model:
                move = self.predict(check_early_win=check_early_win,
                                    prevent_other_win=prevent_other_win)
            else:
                move, p = self.game.smart_action(player, legal_only=legal_only, n=n)
                if not self.game.is_legal_move(move):
                    print("Illegal move from smart player! ", player, p, move)

            print(f"{self.print_player_string(player)} adds to column {move}...")
            self.game.play_move(player, move)
            self._add_state(self.game, player, move)
            player = player * -1

        print(f"{self.print_player_string(self.game.status)} wins!")
        return self.game.status


    ###########################################################################

    def load_existing_model(self, name, basepath="../data/models/"):
        try:
            self.model = load_model(f"{basepath}{name}", compile=True)
            self.model.predict(np.zeros((1, self.board_nodes + 1)))  # Init predictor
        except Exception as e:
            self.model = None
            print(f"Could not load model!\n{e}")
        else:
            self.model.summary()

    def build_network(self):
        """
        Input : self.width * self.height board (42 squares) + player

            https://keras.io/api/layers/normalization_layers/batch_normalization/
            https://keras.io/api/layers/activations/#relu-function

        Output: 2 => [[player_0_prob, player_1_prob]]

        TODO Experiment with other topologies
        """
        print("Building model...")

        self.model = keras.Sequential()

        # Input layer
        self.model.add(Dense(self.board_nodes + 1, input_dim=self.board_nodes + 1))

        # Normalise layer
        # self.model.add(BatchNormalization())

        # One or more large layers
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(256, activation='relu'))
        # self.model.add(Dense(64, activation='relu'))

        # Smaller ending layer
        self.model.add(Dense(self.board_nodes, activation='relu'))
        # self.model.add(Dense(self.game.width, activation='relu'))

        # Output end layer
        self.model.add(Dense(2, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy',  # Or categorical_crossentropy ?
                           optimizer='nadam',  # Or rmsprop?
                           metrics=['accuracy'])

        self.model.summary()

    def prepare_data(self, input_file, train_ratio=0.8):
        print("Preparing data...")

        data = np.load(input_file)

        # TODO Maybe reverse current player?

        Y = np.clip(data[:, -2:-1], 0, 1)   # (winner), clip values to (0, 1)
        X = np.delete(data, -2, 1)          # Drop second to last column, result = (board state, player)

        size = int(train_ratio * X.shape[0])

        X_train, X_test, Y_train, Y_test = X[:size], X[size:], Y[:size], Y[size:]

        print("Data loaded.")
        return (X_train, Y_train), (X_test, Y_test)

    def train(self, train_data, test_data, epochs=10, batch_size=200):
        print("Training model...")

        train_x, train_y = train_data

        self.model.fit(train_x, train_y,
                       validation_data=test_data,
                       shuffle=True,
                       epochs=epochs,
                       batch_size=batch_size)

        self.model.predict(np.zeros((1, self.board_nodes + 1)))  # Init predictor

    def save_model(self, name="trained_1", basepath="../data/models/", save_structure=False):
        # if not name.endswith(".h5"):
        #     name += ".h5"
        print(f"Saving model to: {basepath}{name}...")

        if not os.path.exists(basepath):
            os.makedirs(basepath)

        if save_structure:
            model_json = self.model.to_json()
            with open(f"{basepath}{name}.json", "w") as fp:
                fp.write(model_json)

        save_model(self.model, f"{basepath}{name}")
        print("Saving complete.")

    def _predict_move_probability(self, player, check_early_win=True):
        player_clipped = max(player, 0)  # Clip to (0, 1)

        # Predict chance of winning for each move
        # and return column with highest chance.
        max_probability = (0, 0.0)

        print(self.print_player_string(player) + ": Testing cols: |", end="")

        for i, move in enumerate(range(self.game.width)):
            if not self.game.is_legal_move(move):
                print(f" {colorama.Style.BRIGHT}{colorama.Fore.RED}{0.0:.2f}{colorama.Style.RESET_ALL} |", end="")
                continue

            test_game = Game(board=self.game.board.copy())
            test_game.play_move(player, move)

            if check_early_win and test_game.status == player:
                # Win reached
                print(" win", end="")
                max_probability = (i, 1.0)
                break

            # Get prediction for move
            test_input = np.concatenate((test_game.board.flatten(), [player_clipped])).reshape((1, self.board_nodes + 1))
            prediction = self.model.predict(test_input)[0][player_clipped]  # [[player_0_prob, player_1_prob]]

            if np.isnan(prediction):
                print("Error: prediction is NaN?")
                return 0

            print(f" {prediction:.2f} |", end="")

            if prediction > max_probability[1]:
                max_probability = (i, prediction)

        print(f"  => Predicted move at col {max_probability[0]} with {max_probability[1] * 100:.2f}%")
        return max_probability

    def predict(self, ai_player=-1, check_early_win=True, prevent_other_win=True):
        # Get (move, chance) that AI wins
        ai_move = self._predict_move_probability(ai_player, check_early_win)

        if prevent_other_win:
            other_player_move = self._predict_move_probability(ai_player * -1, check_early_win)

            if other_player_move[1] > ai_move[1]:
                print(f"{colorama.Style.BRIGHT}{colorama.Fore.RED}Trying to prevent{colorama.Style.RESET_ALL}" \
                    + f" {self.print_player_string(ai_player * -1)} from winning...")
                return other_player_move[0]

        return ai_move[0]



def test_accuracy(game, strategy, total=100, **kw):
    win_count, total_games = 0, total

    with HidePrintingContext():
        # Don't print while testing

        for _ in tqdm(range(total_games), ncols=79, desc="Simulating games...", ):
            result = strategy(**kw)
            game.reset()
            if result == -1:
                win_count +=1

    return win_count / total_games


if __name__ == "__main__":
    game = MLC4()

    # data_input = "../data/c4-10k.npy"
    # data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-10k.npy"
    # trained_name = "trained_10k"
    data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-50k.npy"
    trained_name = "trained_50k"
    train_new = False

    if train_new:
        # Build. train and save new model
        train_data, test_data = game.prepare_data(data_input, train_ratio=0.8)
        game.build_network()
        game.train(train_data, test_data, epochs=10, batch_size=200)
        game.save_model(trained_name)
    else:
        # Or load pre-trained
        game.load_existing_model(trained_name)


    input("Model ready. Press enter to start game.")
    check_early_win   = True
    prevent_other_win = True


    # # Test games
    # while True:
    #     total = 100
    #     try:
    #         rnd_acc = test_accuracy(game, game.play_vs_random, total, check_early_win=check_early_win, prevent_other_win=prevent_other_win)
    #         print(f"Accuracy vs random: {rnd_acc * 100:.2f}%")

    #         smt_acc = test_accuracy(game, game.play_vs_smart , total, n=3, check_early_win=check_early_win, prevent_other_win=prevent_other_win)
    #         print(f"Accuracy vs smart : {smt_acc * 100:.2f}%")

    #         input()
    #     except KeyboardInterrupt:
    #         break


    # # Play and display games
    while True:
        game.play_vs_random(check_early_win=check_early_win, prevent_other_win=prevent_other_win)
        # game.play_vs_smart(n=5, check_early_win=check_early_win, prevent_other_win=prevent_other_win)

        print()
        # game.print_states()

        ask = input(f"{colorama.Fore.GREEN}Enter 'q' to exit, else play new game:{colorama.Style.RESET_ALL} ")
        if ask.lower() in ("q", "quit", "exit"):
            break

        game.reset()
