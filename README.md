# Forex Exchange Rates: Time-Series Analysis and ARIMA Forecasting

Analysis of daily exchange rates for 22 currencies against the US dollar from January 2000 to December 2019. The notebook covers data cleaning, year-, month- and day-level visualisation, and one-step-ahead forecasting with ARIMA.

University of Essex, CF969 Big Data for Computational Finance (supervisor Dr Panagiotis Kanellopoulos), April 2020. Individual work.

## What the notebook does

1. **Cleaning:** the dataset marks missing days with `ND`. These are converted to `NaN` and dropped, leaving 5,015 complete trading days across all 22 series. Columns are then cast to numeric.
2. **Exploration:** a full-period plot for each currency, side-by-side plots for 2016, and day-level plots for May 2017.
3. **Forecasting:** ARIMA(2,1,0) with walk-forward validation. The first 70% of the series is the initial history. For each of the remaining ~1,500 days, the model is refitted on all data seen so far and predicts the next day, after which the true value is appended.

## Results (GBP/USD)

From `forex_exchange_analysis_reproducible.ipynb`:

| Forecast | Test MSE | Test RMSE |
|---|---:|---:|
| ARIMA(2,1,0), walk-forward | 1.760 × 10⁻⁵ | 0.00420 |
| Persistence (tomorrow = today) | 1.761 × 10⁻⁵ | 0.00420 |
| Ratio ARIMA / persistence | **0.9997** | |

ARIMA's errors are effectively the same as those of the naive persistence forecast. This is the expected outcome for a liquid currency pair, whose daily changes are close to a random walk. It also explains why the prediction line in the actual-vs-predicted plot shadows the true series with a one-day lag. The original notebook printed MSE to four decimal places (`0.0000`), which hid this comparison. The persistence baseline was added in the reproducible version to make it visible.

## Repository contents

| File | Description |
|---|---|
| `forex_exchange_analysis_new.ipynb` | Original notebook with saved outputs |
| `forex_exchange_analysis_reproducible.ipynb` | Same analysis, updated to run on current libraries, with a persistence baseline (executed, outputs included) |
| `Foreign_Exchange_Rates.csv` | Daily rates, 22 currencies vs. USD, 2000–2019 (Kaggle, "Foreign Exchange Rates 2000–2019") |
| `Big_Data_For_Computational_Finance.pdf` | Part 1 of the coursework: review of *Classification-based Financial Markets Prediction using Deep Neural Networks* (Dixon, Klabjan & Bang) |
| `requirements.txt` | Python dependencies |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook forex_exchange_analysis_reproducible.ipynb
```

Set `CURRENCY` in the forecasting section to any column name, or to `None` to choose interactively. The walk-forward loop refits ARIMA about 1,500 times and takes roughly two minutes on a laptop.

The reproducible copy differs from the original only where current libraries require it:

- `statsmodels.tsa.arima_model.ARIMA` was removed in statsmodels 0.13 and is replaced by `statsmodels.tsa.arima.model.ARIMA`.
- The CSV path pointed to a `forex_exchange_files/` folder that is not in the repository.
- `np.NAN` was removed in NumPy 2.0 (now `np.nan`), and the `seaborn` Matplotlib style was renamed.
- The currency is chosen with a variable rather than an `input()` prompt, so the notebook runs top to bottom.
- Only the first 10 walk-forward steps are printed, instead of all ~1,500.
