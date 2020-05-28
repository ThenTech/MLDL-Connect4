# MLDL-Connect4
2020-06-12 **Cedric Mingneau** (1232611), **William Thenaers** (1746077)

### Base model and tested accuracy

```python
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 43)                1892
_________________________________________________________________
dense_2 (Dense)              (None, 64)                2816
_________________________________________________________________
dense_3 (Dense)              (None, 256)               16640
_________________________________________________________________
dense_4 (Dense)              (None, 256)               65792
_________________________________________________________________
dense_5 (Dense)              (None, 42)                10794
_________________________________________________________________
dense_6 (Dense)              (None, 3)                 86
=================================================================
Total params: 98,020
Trainable params: 98,020
Non-trainable params: 0
```

Training was done on the provided datasets with 15 epochs, `0.85` ratio between training and test data and a learning rate of `0.001` for the [`Nadam`](https://keras.io/api/optimizers/Nadam/) optimizer function. All layers use the [`relu`](https://keras.io/api/layers/activations/#relu-function) activation, except for the last one which uses a [`softmax`](https://keras.io/api/layers/activations/#softmax-function) function. The model was compiled with the [`SparseCategoricalCrossentropy`](https://keras.io/api/losses/probabilistic_losses/#sparsecategoricalcrossentropy-class) as loss function. The standard [`accuracy`](https://keras.io/api/metrics/accuracy_metrics/#accuracy-class) metric was used for evaluation.

Accuracies in tables below are from averages of 1000 simulated games, with player `-1` as the trained network and player `1` as random or smart strategy (from `Game` class). The `vs smart (n=100)` is only 100 games, since it took quite long. All random generators were seeded to `1232611` for these tests. If they are randomly seeded, the performance seems to increase on average. Fitted accuracy column contains the final accuracy achieved on the trainings data set and the evaluated accuracy after fitting the model and passing the test set. Running one row of tests, takes around 25-30 minutes. While training the network with "only" 50k games, averaging at about 15 moves per game (so about 750k boards in total), is fairly fast, playing against the monte-carlo player (the `vs smart` player) is the slowest. With a depth `n = 100`, playing just 10 games against it, takes half the total time.

Two extra options are provided:

- `check_early_win` will always choose the column that would result in a 100% win (4 on a row), instead of the column with the highest chance.
- `prevent_other_win` will choose the best move from the **opponent** if it has a higher chance of winning than any of its own moves, and thus *trying* to prevent the opponent from winning.

The `10k` and `50k` datasets were used with the normal network as seen above. An additional network was created with a normalised board input (42 instead of 43 input nodes). In this case the board is converted so that it's always seen from the point of view of the current player. The models trained with this adjustment, have the `-norm` suffix.

The results:

| Source training | Fitted accuracy | Test vs random | vs smart (n=3) |  (n=5)  | (n=100) |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 10k | 81.00/59.51% | 96.60% | 79.20% | 67.90% |  4.00%  |
| 10k-norm | 75.74/62.09% | 34.90% |     7.50%      | 2.50%  | 0.00% |
| 50k |  74.98/62.66%   | 98.60% | 88.00% | 82.00% | 13.00% |
| 50k-norm | 71.48/64.51% | 28.10% | 5.30% | 2.70% | 0.00% |
(**with** `check_early_win` and **without** `prevent_other_win`)

| Source training | Fitted accuracy | Test vs random | vs smart (n=3) |  (n=5)  | (n=100) |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 10k | 81.00/59.51% | 98.30% | 88.00% | 79.00% | 13.00% |
| 10k-norm | 75.74/62.09% | 93.20% | 71.00% | 55.00% | 5.00% |
| 50k | 74.98/62.66% | 99.50% | 96.50% | 90.40% | 18.00% |
| 50k-norm | 71.48/64.51% | 95.40% | 79.30% | 66.10% | 8.00% |
(**with** `check_early_win` and **with** `prevent_other_win`)

### New data generation

We used the above model to generate some novel data, by letting the created AI play against the smart player `n=10`. We used the model trained against the 50k dataset, since this had the best performance. Playing games with a `ProcessPoolExecutor` and using only the CPU to run the pre-trained model showed to be the fastest. Simulating these games took 31 minutes and 18 seconds for 10k and 2 hours 41 minutes for 50k games. Additionally, we merged the given 50k dataset with the newly generated one, creating a merged 100k dataset.

New models were trained using these datasets. The results using the random seed mentioned before, are shown below.

| Source training | Fitted accuracy | Test vs random | vs smart (n=3) |  (n=5)  | (n=100) |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 10k AI vs smart | 85.69/72.53% | 97.50% | 87.50% | 76.90% | 9.00% |
| 50k AI vs smart | 82.64/74.68% | 98.20% | 88.70% | 84.40% | 15.00% |
| 100k merged | 75.92/73.90% | 99.70% | 97.30% | 93.10% | 22.00% |

(**with** `check_early_win` and **with** `prevent_other_win`)

We can see the accuracy against its own training and test data is much higher than those from before, but the new 10k and 50k models do not perform any better than the previous ones. The 100k model does perform marginally better however. This is likely due to the fact that double the amount of data was used to train it.