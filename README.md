# MLDL-Connect4

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
dense_6 (Dense)              (None, 2)                 86
=================================================================
Total params: 98,020
Trainable params: 98,020
Non-trainable params: 0
```

Accuracies in tables below are from averages of 1000 simulated games, with player `-1` as the trained network and player `1` as random or smart strategy (from `Game` class). The `vs smart (n=100)` is only 100 games, since it took quite long.

| Source training | Fitted accuracy | Test vs random | Test vs smart (n=3) |  (n=5)  | (n=100) |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 10k | 80.76% |   |   |   |   |
| 50k | 74.06% | 95-99% | ~83-88% | ~76-80% | ~6% |
(**with** `check_early_win` and **without** `prevent_other_win`)

| Source training | Fitted accuracy | Test vs random | Test vs smart (n=3) |  (n=5)  | (n=100) |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| 10k | 80.76% |   |   |   |   |
| 50k | 74.06% | 98-100% | ~89-97% | ~84-92% | ~10% |
(**with** `check_early_win` and **with** `prevent_other_win`)
