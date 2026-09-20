# Tennis Match Winner Prediction

This project builds and compares machine learning models to predict the winner of professional tennis matches using historical ATP-style match data, player statistics, and bookmaker odds.

The workflow includes data cleaning, feature engineering, temporal train/test validation, model comparison, and evaluation through standard classification metrics such as accuracy and AUC-ROC.

## Project goal

The main objective is to estimate whether Player 1 or Player 2 will win a match, framed as a binary classification problem:

- Target label `y = 1`: Player 1 wins
- Target label `y = 0`: Player 2 wins

The project explores whether combining traditional tennis indicators (rankings, points, surface form, head-to-head behavior, recent form) with bookmaker odds produces a robust predictive model.

## Data

The project expects annual Excel files stored in a `data/` folder, for example:

- `data/2019.xlsx`
- `data/2020.xlsx`
- `data/2021.xlsx`
- `data/2022.xlsx`
- `data/2023.xlsx`
- `data/2024.xlsx`
- `data/2025.xlsx`

The notebooks load all available years and concatenate them into a single dataset.

The raw dataset contains match-level information such as:

- players
- tournament details
- venue and surface
- round and event category
- rankings and rating points
- historical bookmaker odds
- match result and metadata

## Pipeline overview

The pipeline is implemented in the notebook `main.ipynb` and supported by the helper modules:

- `config.py`: global settings, feature ordering, ordinal mappings, and column definitions
- `features.py`: custom feature engineering logic
- `models.py`: model evaluation, summary reporting, ROC plotting, and error analysis

### 1. Data loading and inspection

The project loads yearly Excel files and concatenates them into one timeline of tennis matches. The code checks the total shape of the dataset and inspects column types and missing values.

### 2. Data preparation

This phase includes:

- removing incomplete matches (`Comment != Completed`)
- dropping irrelevant or leakage-prone columns
- handling missing values with imputation rules
- randomizing player-side assignment for modeling purposes
- creating synthetic `Player1`/`Player2` roles and deriving `P1Rank`, `P2Rank`, `P1Pts`, `P2Pts`, and average odds for each side
- encoding categorical variables such as `Surface`, `Round`, and `Series`

### 3. Feature engineering

The custom feature generation includes these historical signals:

- overall win rate per player
- surface-specific win rate per player
- recent form over the last 20 matches
- head-to-head win rate
- differential variables such as:
  - `rank_diff`
  - `pts_diff`
  - `win_rate_diff`
  - `win_rate_surface_diff`
  - `recent_win_rate_diff`
  - `h2h_win_rate_diff`

These features are engineered so that the model works with pairwise differences between players rather than raw absolute values.

### 4. Temporal split

The project uses a time-based validation strategy:

- training years: 2019–2024
- test year: 2025

This is important because tennis results are time-dependent and future leakage should be avoided.

## Baseline models

Before advanced models, two simple baselines are evaluated:

1. Ranking baseline
   - predicts that the lower-ranked player wins
2. Bookmaker odds baseline
   - predicts that the player with the lower average odds wins

These baselines help understand whether the ML models truly add predictive value beyond obvious signals.

## Machine learning models

The project trains and compares multiple models.

### XGBoost

An XGBoost classifier is tuned with randomized hyperparameter search and evaluated with:

- accuracy
- ROC-AUC
- classification report
- RFECV feature selection
- comparison with and without odds

The model is saved in the `models/` folder as:

- `xgb_best.pkl`
- `xgb_final.pkl`
- `xgb_no_odds.pkl`
- `xgb_features.json`

### Random Forest

A Random Forest classifier is also tuned and evaluated in the same way as XGBoost, with feature selection and no-odds comparison.

Saved artifacts:

- `rf_best.pkl`
- `rf_final.pkl`
- `rf_no_odds.pkl`
- `rf_features.json`

### Artificial Neural Network (ANN)

Several neural network configurations are trained with TensorFlow/Keras:

- `ann_v1.keras`
- `ann_v2.keras`
- `ann_no_odds.keras`

The ANN models use:

- standardized input features
- binary cross-entropy loss
- sigmoid output layer
- early stopping to avoid overfitting

## Model evaluation

The project computes and prints:

- training and test accuracy
- training and test ROC-AUC
- classification report for the test set
- feature importance rankings
- RFECV final feature subsets
- model comparison via ROC curves
- confidence/error analysis

The project also includes a summary table comparing all models by test accuracy and AUC-ROC.

## Error analysis

The notebook includes a dedicated section for model inspection:

- ROC comparison across models
- confidence distribution
- accuracy by surface
- accuracy by tournament stage
- accuracy by ranking gap

This helps evaluate where the model is strong and where it struggles.

## Repository structure

```text
tennis_prediction/
├── config.py
├── features.py
├── models.py
├── main.ipynb
├── exploration.ipynb
├── README.md
├── models/
│   ├── ann_no_odds.keras
│   ├── ann_v1.keras
│   ├── ann_v2.keras
│   ├── rf_features.json
│   ├── xgb_features.json
│   └── ...
└── data/
    └── (expected yearly Excel files)
```

## Requirements

The project depends on Python packages such as:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- tensorflow
- joblib
- openpyxl (for reading Excel files)

A typical environment can be set up with:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib tensorflow joblib openpyxl
```

## How to run

From the project directory:

```bash
python -m jupyter notebook
```

Then open `main.ipynb` and run the cells in order.

If you prefer a script-based workflow, the project is designed around the notebook pipeline, so the notebook remains the main execution entry point.

## Notes

- This project does not use external APIs; it relies on local historical data files.
- The feature engineering is tuned specifically for tennis match prediction and uses pairwise player differences.
- The train/test split is temporal rather than random, which is more realistic for sports forecasting.
- Bookmaker odds are treated as a strong feature, but the project also evaluates no-odds versions of the models to test out-of-sample performance without market information.

## Potential extensions

Possible future improvements include:

- adding more seasons and tournament history
- testing additional models such as LightGBM, CatBoost, or stacked ensembles
- using more detailed player-level time-series features
- adding calibration methods for probability estimates
- building a reproducible training script separate from the notebook
- creating a small inference API for predicting new matches

## Summary

This repository demonstrates a complete machine learning workflow for tennis match prediction: loading historical match data, engineering meaningful tennis-specific features, benchmarking baselines, training multiple predictive models, and evaluating them with robust, real-world validation.

It is a practical example of sports analytics and binary classification in a structured tabular setting.
