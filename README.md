# Predicting 10-minute returns from a leading instrument

Forecasting the price change of an instrument Y over the next 10 minutes, using 10-second price bars of Y and a
related instrument X (2013 data). Forecasts use only data up to the current bar, and are scored by mean squared
error against the `returns` column.

- [`solution.py`](solution.py): the model. `modelEstimate(train_file)` fits it and `modelForecast(test_file, params)` predicts.
- [`analysis.ipynb`](analysis.ipynb): exploratory analysis and model development.

The data files (`train.csv`: January to September 2013; `test.csv`: October to December 2013) are not included.
Each has columns `timestamp` (ms), `xprice`, `yprice` and `returns`.

## Results

| | MSE improvement over predicting zero |
|---|---|
| Out of sample, April to December 2013 (refit monthly) | 0.45% (95% interval 0.10% to 0.81%) |
| `test.csv` | 0.88% (correlation 0.099) |

Small, as expected for 10-minute-ahead returns. The improvement over predicting zero equals R-squared measured against
zero, so the out-of-sample R-squared is about 45 basis points.

## Model

The idea is that X leads Y: when X moves, Y tends to follow over the next few minutes, and when the gap between the
two prices is unusually wide, it tends to close. Y's own history carries no signal (its past moves have correlations
below 0.015 with its next 10-minute return).

There are two features, both built only from data up to time t:

- **`x_mom`**: an exponentially weighted mean of X's 10-second price changes.
  - It restarts at zero at each session open. The first change of a session spans the overnight break, and Y has
    moved overnight too, so that jump says nothing about the next 10 minutes.
  - It is divided by X's volatility over the last hour. In volatile periods raw momentum is large but mostly noise,
    and without this step the predictions get too big when they are least reliable.
  - The half-life is chosen from 2 to 64 bars by walk-forward validation. It comes out at 48 bars (about 8 minutes),
    so it picks up X's moves over the past several minutes.
- **`spread_z`**: how far the X - Y spread is from its 10-minute average, measured in standard deviations. A large
  value means the gap is likely to narrow.

Both features are standardised and fitted with Ridge regression. The half-life and the Ridge penalty are picked inside
`modelEstimate()` by walk-forward validation on the training file. Finally the predictions are multiplied by 0.8.

Fitted on `train.csv`, the model works out to roughly:

```
predicted return = -0.0005 + 0.080 * x_mom + 0.0039 * spread_z
```

### Why multiply by 0.8

If the predictions are scaled by c, the MSE gain over predicting zero is

```
gain(c) = 2c * cov(prediction, actual) - c^2 * var(prediction)
```

The best c is cov / var, the slope from regressing actual returns on the predictions. If the raw predictions were
sized correctly, that slope would be 1. On data the model hasn't seen, the slope comes out at about 0.8. The signal is
weak and its strength drifts, so full-size predictions are slightly too big on average and can lose to zero in weak
periods. Shrinking a little protects those periods and costs almost nothing in the strong ones.

## Validation

I split `train.csv` by time into a training part and a validation part, chose features and settings on that split,
and used `test.csv` as the test set.

- **Walk-forward validation:** from April to September 2013 the model was refit each month on all earlier data and
  scored on the next month. This gives several out-of-sample months instead of a single score.
- **Feature selection:** a candidate feature was kept only if its walk-forward MSE improvement had a t-statistic above
  3.16 and held up in most months. The usual bar of 2 is too lenient here, because trying many variants of a feature
  and keeping the best one makes t-statistics look better than they are. To set a fair bar, I ran the same search on
  features built from scrambled data, which carry no real information, and took the value that only 5% of those
  reached by chance, which was 3.16.
- **Placebo check:** I reran the whole pipeline 200 times with X's price path swapped for another day's, so X carries
  no real information. The best placebo run beat zero by 0.06%, against 0.45% on the real data, so the model is
  predicting better than noise.

## Other approaches tried

None of these beat the simpler model reliably.

- **Gradient-boosted trees:** lost to Ridge, both on 20 features and on just the two final features. They mainly
  picked up time of day and weekday, which do not carry over to new periods.
- **More lags and windows of X's moves**, including fast plus slow momentum: the extra features overlap heavily and
  mostly add noise.
- **Squares, interaction terms and separate slopes for up and down moves:** no gain. Average returns increase steadily
  and roughly linearly with each feature, so a linear model already captures the shape.
- **Time of day, calendar effects, distance from the day's open and raw price levels:** these fitted patterns specific
  to one period that did not repeat.
- **Other spread definitions** (price units, rolling or fixed hedge ratios): none beat the 10-minute z-score.
- **Fast momentum weighted by how long Y's price has been unchanged:** the most promising idea, but it did not make
  the final features.

  ```
  fast_stale_t = fast_mom_t * min(k_t, 6) / 6
  ```

  Here `fast_mom_t` is `x_mom` with a 2-bar half-life, and `k_t` is the number of bars since Y's price last changed.
  It pointed the right way on the training data but stayed below the adoption bar there, so I left it out.

## Running it

```python
from solution import modelEstimate, modelForecast

params = modelEstimate("train.csv")
predictions = modelForecast("test.csv", params)
```

Fitting takes about 7 seconds. Tested with pandas 1.0 and 3.1.
