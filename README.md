# Bayesian Robot Navigator

An explainable model that reads four ultrasonic distance sensors on a wall-following robot and says what the robot should do next, along with the probability of every possible action.

**Live demo:** (https://prince01104.github.io/Bayesian-Robot-Navigator-/)

![accuracy](https://img.shields.io/badge/test%20accuracy-95.89%25-brightgreen) ![model](https://img.shields.io/badge/model-Bayesian%20network-blue) ![data](https://img.shields.io/badge/data-UCI%20Wall--Following-orange)

## Why this project

A neural network usually returns a single answer. A robot that moves in the real world is easier to trust when it can also say how sure it is and which sensor drove the decision. A Bayesian network gives both, and it does so with small probability tables that a person can read.

## Dataset

[Wall-Following Robot Navigation Data](https://archive.ics.uci.edu/dataset/194/wall%2Bfollowing%2Brobot%2Bnavigation%2Bdata) from the UCI Machine Learning Repository (Freire, Veloso and Barreto, CC BY 4.0).

- A SCITOS G5 robot drives around a room following the wall, sampled 9 times per second.
- This project uses `sensor_readings_4.data`: 5,456 readings of four minimum distances (front, left, right, back).
- Four actions: `Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn`, `Slight-Left-Turn`.
- No missing values.

## Method

1. **Chronological split.** The first 75% of the rows are used for training and the last 25% (1,364 rows) for testing. Neighbouring readings are almost identical, so a random split would leak information into the test set.
2. **Discretisation.** Each distance is cut into quantile bins. The bin edges are computed from the training rows only.
3. **Bayesian network.** One `Action` node is the parent of four sensor nodes. Given the action, the sensors are treated as independent (naive Bayes structure).
4. **Learning.** The prior `P(action)` and the tables `P(bin | action)` for each sensor are counted from the training data, with Laplace smoothing so no probability is ever zero.
5. **Inference.** For each action, multiply the prior by the likelihood of each sensor's bin, then normalise:

   ```
   P(action | sensors) ∝ P(action) × Π P(bin_sensor | action)
   ```

   Calculations are done in log space for numerical stability.
6. **Choosing the number of bins.** The number of bins was compared on a validation set (the last 20% of the training rows). The test set was used once, at the end.

## Results

All numbers below are on the 1,364 held-out test readings.

| Model | Accuracy |
|---|---|
| Always predict the most common action | 40.62% |
| Bayesian network, 3 bins | 90.18% |
| **Bayesian network, 10 bins (used in the demo)** | **95.89%** |

### Number of bins

| Bins | 3 | 5 | 7 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|
| Validation accuracy | 85.47% | 89.87% | 92.80% | 94.99% | 94.87% | 94.87% | 96.46% |

Validation accuracy rose quickly up to 10 bins and then flattened. 30 bins scored best on validation (96.77% on the test set), but the gain over 10 bins is about 1.5 points on roughly 820 rows, so 10 bins was kept for a smaller and more readable model.

### Confusion matrix (10 bins, rows = actual, columns = predicted)

| | Move forward | Sharp right | Slight left | Slight right | Recall |
|---|---|---|---|---|---|
| **Move forward** | 510 | 17 | 6 | 21 | 92% |
| **Sharp right** | 6 | 522 | 0 | 0 | 99% |
| **Slight left** | 0 | 1 | 106 | 0 | 99% |
| **Slight right** | 5 | 0 | 0 | 170 | 97% |
| **Precision** | 98% | 97% | 95% | 89% | |

Most mistakes are `Move-Forward` readings that the model classes as a slight turn. Those cases sit close to the boundary between going straight and correcting course.

### Which sensor matters (10 bins)

| Sensor removed | Test accuracy |
|---|---|
| None | 95.89% |
| Front | 69.13% |
| Left | 88.71% |
| Right | 96.26% |
| Back | 95.97% |

The front sensor is essential and the left sensor is useful. Removing the right or back sensor changes accuracy by less than half a point, which is within noise for 1,364 rows.

## Web demo

`index.html` is a single file with the trained model embedded. It needs no server and no build step.

- **Explorer:** move four sliders and watch the decision, the probability of each action and a per-sensor explanation update. Any sensor can be switched off to see how the model copes with a failure.
- **Replay:** play back the 1,364 test readings. The page shows the actual action next to the model's answer and keeps a running accuracy.
- **Insights:** confusion matrix, sensor importance, the bin comparison and a heatmap of the learned probability tables. These are computed in the browser from the embedded test data.

To run it locally, open `index.html` in a browser. To publish it, push this repository to GitHub and enable **Settings → Pages → Deploy from branch**.

## Repository structure

```
.
├── index.html        # web demo (model and test data embedded)
├── model.json        # prior, likelihood tables and bin edges
├── notebook.ipynb    # data loading, training and evaluation
└── README.md
```

`model.json` keys: `sensors`, `actions`, `n_bins`, `cuts` (bin edges per sensor), `prior`, `likelihood[sensor][action][bin]`.

## Reproducing the results

```bash
pip install pandas numpy scikit-learn matplotlib
```

Download `sensor_readings_4.data` from the UCI page above, place it next to the notebook and run all cells. The data has no header; the columns are `front, left, right, back, action`.

## Limitations

- The naive Bayes structure assumes the sensors are independent given the action. Real sensors are correlated, which can make the model overconfident.
- The data comes from one robot in one room. Accuracy in a different environment is unknown.
- Bin edges come from quantiles, so they depend on the distribution of the training run.
- The test set is one continuous run, so the 1,364 readings are strongly correlated and the true uncertainty on the accuracy is larger than the sample size suggests.
- This is a learning project, not a controller for a real robot.

## Possible extensions

- Add the previous reading as a node (a dynamic Bayesian network) so the model can use short-term memory.
- Compare against a small MLP or LSTM on the same split.
- Check calibration: when the model says 80%, is it right 80% of the time?
- Use the 24-sensor version of the data with feature selection.

## Credits

Dataset: A. L. Freire, G. A. Barreto, M. Veloso and A. T. Varela, "Short-term memory mechanisms in neural network learning of robot navigation tasks: A case study", 2009. Distributed through the UCI Machine Learning Repository under CC BY 4.0.

## License

Code: MIT (change if you prefer another). Data: CC BY 4.0, see the credit above.
