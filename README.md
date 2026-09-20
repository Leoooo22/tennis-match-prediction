# Tennis Match Winner Prediction

This project predicts the winner of a tennis match using historical ATP-style data, player statistics, bookmaker odds, and machine learning models. The main goal is to learn whether Player 1 or Player 2 is more likely to win a given match, modeled as a binary classification problem.

## Project overview

The repository contains a complete pipeline for:

- loading historical match data from Excel files
- cleaning and preparing the dataset
- engineering tennis-specific features
- training and comparing multiple machine learning models
- evaluating test performance with accuracy, AUC-ROC, and error analysis

The analysis is implemented mainly in the notebook `main.ipynb`, while reusable logic is split across:

- `config.py`: feature config, default mappings, and column definitions
- `features.py`: feature engineering functions
- `models.py`: evaluation utilities, ROC plotting, and classification reporting

## Repository structure

```text
tennis_prediction/
├── README.md
├── Relazione.pdf
├── config.py
├── features.py
├── models.py
├── main.ipynb
├── exploration.ipynb
├── requirements.txt
├── data/
│   ├── 2019.xlsx
│   ├── 2020.xlsx
│   ├── 2021.xlsx
│   ├── 2022.xlsx
│   ├── 2023.xlsx
│   ├── 2024.xlsx
│   └── 2025.xlsx
├── models/
│   ├── ann_no_odds.keras
│   ├── ann_v1.keras
│   ├── ann_v2.keras
│   ├── rf_features.json
│   ├── xgb_features.json
│   └── ...
└── .git/
```

## Data

The dataset is stored locally in the `data/` folder as yearly `.xlsx` files. Each file contains historical tennis match information such as:

- players
- tournament, date, round, surface, and court
- rankings and points
- bookmaker odds
- match winner and result metadata

The project loads all selected years and combines them into a single dataframe for training and evaluation.

## Data preparation pipeline

The workflow includes several steps:

1. Load annual Excel files and concatenate them
2. Remove incomplete or irrelevant matches
3. Drop leakage-prone columns
4. Handle missing values with imputation and indicator flags
5. Randomize the side assignment of players to create a consistent `Player1` / `Player2` structure
6. Encode categorical variables such as round, surface, and series
7. Create engineered historical features
8. Split data temporally into train and test sets

## Feature engineering

The custom feature generation includes:

- overall win rate
- surface-specific win rate
- recent form over the last 20 matches
- head-to-head win rate
- differential features such as:
  - `rank_diff`
  - `pts_diff`
  - `win_rate_diff`
  - `win_rate_surface_diff`
  - `recent_win_rate_diff`
  - `h2h_win_rate_diff`

These variables are then used to model the probability that Player 1 wins the match.

## Modeling approach

The project benchmarks several models:

### Baselines

- Ranking-based baseline: lower-ranked player wins
- Odds-based baseline: player with lower average odds wins

### Tree-based models

- XGBoost
- Random Forest

Both are tuned using cross-validation and evaluated with:

- accuracy
- ROC-AUC
- classification report
- feature importance analysis
- RFECV feature selection
- comparison with and without bookmaker odds

### Neural network models

- ANN v1
- ANN v2
- ANN without odds

These are trained with TensorFlow/Keras and standardized inputs using `StandardScaler`.

## Evaluation strategy

The project uses a temporal split:

- training: years 2019–2024
- test: year 2025

This avoids leakage from future information and better reflects real-world forecasting conditions.

Models are evaluated using:

- accuracy
- AUC-ROC
- classification report
- ROC curves for model comparison
- confidence and error analysis

## Requirements

The project dependencies are listed in `requirements.txt` and include:

```text
numpy
pandas
openpyxl
scikit-learn
xgboost
tensorflow
shap
matplotlib
joblib
seaborn
ipykernel
```

To install them:

```bash
pip install -r requirements.txt
```

## How to run

From the project folder:

```bash
jupyter notebook
```

Then open `main.ipynb` and run the cells in order.

If you want to reproduce the environment from scratch, make sure the `data/` folder contains the yearly Excel files before running the notebook.

## Notes

- The project relies on local data files and does not use a remote API.
- The models are trained with a realistic time-based split, which is important in sports prediction.
- Bookmaker odds are treated as an informative signal, but the project also evaluates versions without odds to study their standalone predictive value.
- The trained model artifacts are stored in `models/` for reuse and comparison.

## Summary

This repository implements a machine learning workflow for tennis match prediction based on historical tennis data and engineered features. It combines classical statistical signals with modern predictive models to compare different strategies and evaluate which approach generalizes best on future matches.

The project is suitable for both experimentation and academic reporting, and it includes both notebook-based analysis and trained model artifacts for reuse.
