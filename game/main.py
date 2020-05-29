from mlc4 import MLC4, MLC4Normalised, MLC4NormalisedConv, MLC4Simplified
from colours import Colours, style

import sys
from io import StringIO
from tqdm import tqdm
import concurrent.futures

class HidePrintingContext:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def test_accuracy(game, strategy, total=100, **kw):
    # Returns (win %, draw %, loss %)
    counts = {
        -1 : 0,
         0 : 0,
         1 : 0
    }

    with HidePrintingContext():
        # Don't print while testing

        for game_id in tqdm(range(total), ncols=79, desc="Simulating games..."):
            result = strategy(**kw)
            game.reset()
            counts[result] += 1

    return tuple(map(lambda x: x / total, counts.values()))


def _setup_and_play_mp(arg_tup):
    game_id, klass, model, kw = arg_tup
    new_game = klass()
    new_game.model = model

    with HidePrintingContext():
        new_game.play_vs_smart(**kw)

    io = StringIO()
    new_game.print_states(game_id, io)
    return io.getvalue()

def generate_dataset(game, output_file_name, total=100, mode="w", **kw):
    if output_file_name:
        output_file = open(output_file_name, mode)
    else:
        print(style("No output file?", Colours.FG.BRIGHT_RED))
        return

    with HidePrintingContext():
        # Concurrently (for total=100, takes around 36 seconds)
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
            for result in tqdm(pool.map(_setup_and_play_mp,
                                        ((gid, game.__class__, game.model, kw) for gid in range(total)),
                                        chunksize=50),
                               ncols=79, desc="Generating data... ", total=total,):
                print(result, file=output_file)

        # Sequentially (for total=100, takes around 1 minute)
        # for gid in tqdm(range(total), ncols=79, desc="Generating data... ", total=total):
        #     result = setup_and_play(gid)
        #     print(result, file=output_file)

    if output_file_name:
        print(style("Data saved to: '", Colours.FG.GREEN) + style(output_file_name, Colours.FG.BRIGHT_GREEN) + "'")
        output_file.close()


def print_accuracy(descr, win=0, draw=0, loss=0, result=None):
    if result is not None:
        win, draw, loss = result

    print(f"{descr}: " \
        + style(f"{win * 100:.2f}% wins", Colours.FG.GREEN)    + ", " \
        + style(f"{draw * 100:.2f}% draws", Colours.FG.YELLOW) + ", " \
        + style(f"{loss * 100:.2f}% losses", Colours.FG.RED)   + ".")


if __name__ == "__main__":
    # Create one of these two networks.
    # game = MLC4()
    # game = MLC4Normalised()
    # game = MLC4NormalisedConv()
    game = MLC4Simplified()

    # data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-10k.npy"
    # trained_name = "trained_10k"
    # trained_name = "trained_10k-norm"

    # data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-50k.npy"
    # trained_name = "trained_50k"
    # trained_name = "trained_50k-norm"

    # data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-ai-vs-smart-10k.npy"
    # trained_name = "c4-ai-vs-smart-10k"

    # data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-ai-vs-smart-50k.npy"
    # trained_name = "c4-ai-vs-smart-50k"

    data_input   = "G:\\_temp\\UHasselt\\MLDL\\c4-merge-100k.npy"
    # trained_name = "c4-merge-100k"
    trained_name = "trained_100k-simple"


    train_new = False

    if train_new:
        # Build, train and save new model
        train_data, test_data = game.prepare_data(data_input, train_ratio=0.85)
        game.build_network(trained_name)
        game.train(train_data, test_data, epochs=15, batch_size=200,
                   show_plot=True, save_plot_path=f"../data/models/{trained_name}.png")
        game.save_model(trained_name)
    else:
        # Or load pre-trained
        game.load_existing_model(trained_name)

    if not game.has_model():
        print(style("No model ready! Exiting...", Colours.FG.BRIGHT_RED))
        exit(1)


    check_early_win   = True
    prevent_other_win = True

    if any((check_early_win, prevent_other_win)):
        print(style("Using options: ", Colours.FG.MAGENTA) \
            + style("check_early_win", Colours.FG.BRIGHT_GREEN if check_early_win else Colours.FG.BRIGHT_BLACK) + ", "\
            + style("prevent_other_win", Colours.FG.BRIGHT_GREEN if prevent_other_win else Colours.FG.BRIGHT_BLACK))

    input(style("Model ready. Press enter to start tests or play.", Colours.FG.MAGENTA))


    # Test games
    run_test_or_generate = False
    run_play_and_display = True

    output_file = ""
    # output_file = "../data/c4-ai-vs-smart-10k.csv"
    # output_file = "../data/c4-ai-vs-smart-50k.csv"

    if run_test_or_generate:
        if output_file:
            # Create new dataset

            # When using this, also uncomment in mlc4.py:
            #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            # This way the CPU is used to run the network, enabling multiprocessing.

            generate_dataset(game, output_file,
                            total=50000, mode="w",
                            n=10, check_early_win=check_early_win, prevent_other_win=prevent_other_win)
            exit(0)
        else:
            # Test procedure to get accuracies by simulating games
            try:
                result = test_accuracy(game, game.play_vs_random, 1000,
                                    check_early_win=check_early_win, prevent_other_win=prevent_other_win)
                print_accuracy("Accuracy vs random (x1000)     ", result=result)

                result = test_accuracy(game, game.play_vs_smart, 1000, n=3,
                                    check_early_win=check_early_win, prevent_other_win=prevent_other_win)
                print_accuracy("Accuracy vs smart (x1000, n=3) ", result=result)

                result = test_accuracy(game, game.play_vs_smart, 1000, n=5,
                                    check_early_win=check_early_win, prevent_other_win=prevent_other_win)
                print_accuracy("Accuracy vs smart (x1000, n=5) ", result=result)

                result = test_accuracy(game, game.play_vs_smart, 100, n=100,
                                    check_early_win=check_early_win, prevent_other_win=prevent_other_win)
                print_accuracy("Accuracy vs smart (x100, n=100)", result=result)

                # result = test_accuracy(game, game.play_vs_ai, 1000,
                #                        check_early_win=check_early_win, prevent_other_win=prevent_other_win,
                #                        random_move_chance=0.3)
                # print_accuracy("Accuracy vs AI (x1000, rnd=.3) ", result=result)
            except KeyboardInterrupt:
                pass

        input(style("Done. Press enter to play and display games.", Colours.FG.MAGENTA))

    # Play and display games
    while run_play_and_display:
        try:
            # game.play_vs_random(check_early_win=check_early_win, prevent_other_win=prevent_other_win)
            game.play_vs_smart(n=100, check_early_win=check_early_win, prevent_other_win=prevent_other_win)
            # game.play_vs_ai(check_early_win=check_early_win, prevent_other_win=prevent_other_win, random_move_chance=0.3)

            print()
            # game.print_states()

            ask = input(style("Enter 'q' to exit, else play new game: ", Colours.FG.GREEN))
            if ask.lower() in ("q", "quit", "exit"):
                break
        except KeyboardInterrupt:
            break
        finally:
            game.reset()
