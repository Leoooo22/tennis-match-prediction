import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.base import clone
from sklearn.feature_selection import RFECV

from config import ODDS_COLS

def evaluate_model(model, param_grid, X_train, X_test, y_train, y_test, model_name="Model"):

    results = {}

    # 1. Initial training
    print(f"\n{'='*50}")
    print(f"{model_name} — Initial parameters")
    print(f"{'='*50}")

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    results["Initial"] = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "AUC-ROC": roc_auc_score(y_test, y_prob_test),
    }

    print("=== TRAINING ===")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_train, y_prob_train):.4f}")
    print("=== TEST ===")
    print(f"Accuracy: {results['Initial']['Accuracy']:.4f}")
    print(f"AUC-ROC:  {results['Initial']['AUC-ROC']:.4f}")
    print("=== CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(y_test, y_pred_test))

    # 2. Hyperparameter tuning
    print(f"\n{'='*50}")
    print(f"{model_name} — Hyperparameter tuning")
    print(f"{'='*50}")

    tscv = TimeSeriesSplit(n_splits=5)
    tuned = RandomizedSearchCV(
        model, param_grid,
        n_iter=30, cv=tscv,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )
    tuned.fit(X_train, y_train)

    print(f"Best Score: {tuned.best_score_:.4f}")
    print(f"Best Params: {tuned.best_params_}")

    best = tuned.best_estimator_

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)
    y_prob_train = best.predict_proba(X_train)[:, 1]
    y_prob_test = best.predict_proba(X_test)[:, 1]

    results["Tuned"] = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "AUC-ROC":  roc_auc_score(y_test, y_prob_test),
    }

    print("=== TRAINING ===")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_train, y_prob_train):.4f}")
    print("=== TEST ===")
    print(f"Accuracy: {results['Tuned']['Accuracy']:.4f}")
    print(f"AUC-ROC:  {results['Tuned']['AUC-ROC']:.4f}")
    print("=== CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(y_test, y_pred_test))

    # 3. Feature importance
    print(f"\n{'='*50}")
    print(f"{model_name} — Feature Importance")
    print(f"{'='*50}")

    importances = best.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(importance_df.to_string(index=False))

    plt.figure(figsize=(12, 8))
    sorted_idx = importances.argsort()[::-1]
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), X_train.columns[sorted_idx], rotation=90)
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.show()

    # 4. RFECV
    print(f"\n{'='*50}")
    print(f"{model_name} — RFECV Feature Selection")
    print(f"{'='*50}")

    selector = RFECV(
        estimator=best,
        step=1,
        cv=tscv,
        scoring="accuracy",
        min_features_to_select=1,
    )
    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.support_].tolist()
    print(f"Optimal number of features: {selector.n_features_}")
    print(f"Selected features: {selected_features}")

    # 5. Retraining with selected features
    print(f"\n{'='*50}")
    print(f"{model_name} — Final model with selected features")
    print(f"{'='*50}")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    final_model = clone(tuned.best_estimator_)
    final_model.fit(X_train_sel, y_train)

    y_pred_final = final_model.predict(X_test_sel)
    y_prob_final = final_model.predict_proba(X_test_sel)[:, 1]

    results["RFECV"] = {
        "Accuracy": accuracy_score(y_test, y_pred_final),
        "AUC-ROC":  roc_auc_score(y_test, y_prob_final),
    }

    print(f"Final accuracy: {results['RFECV']['Accuracy']:.4f}")
    print(f"Final AUC-ROC:  {results['RFECV']['AUC-ROC']:.4f}")

    # 6. Without odds
    print(f"\n{'='*50}")
    print(f"{model_name} — Without bookmaker odds")
    print(f"{'='*50}")

    X_train_no_odds = X_train.drop(columns=ODDS_COLS)
    X_test_no_odds = X_test.drop(columns=ODDS_COLS)

    model_no_odds = clone(tuned.best_estimator_)
    model_no_odds.fit(X_train_no_odds, y_train)

    y_pred_no_odds = model_no_odds.predict(X_test_no_odds)
    y_prob_no_odds = model_no_odds.predict_proba(X_test_no_odds)[:, 1]

    results["No odds"] = {
        "Accuracy": accuracy_score(y_test, y_pred_no_odds),
        "AUC-ROC":  roc_auc_score(y_test, y_prob_no_odds),
    }

    print(f"Accuracy without odds: {results['No odds']['Accuracy']:.4f}")
    print(f"AUC-ROC without odds:  {results['No odds']['AUC-ROC']:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"{model_name} — Summary")
    print(f"{'='*50}")
    print(f"{'Variant':<12} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("-" * 34)
    for variant, metrics in results.items():
        print(f"{variant:<12} {metrics['Accuracy']:>10.4f} {metrics['AUC-ROC']:>10.4f}")

    return best, selected_features, final_model, model_no_odds

def report_model(best, selected_features, final_model, model_no_odds, X_train, X_test, y_train, y_test, model_name="Model"):

    results = {}

    # 1. Tuned model
    print(f"\n{'='*50}")
    print(f"{model_name} — Tuned model")
    print(f"{'='*50}")

    y_pred_train = best.predict(X_train)
    y_pred_test = best.predict(X_test)
    y_prob_train = best.predict_proba(X_train)[:, 1]
    y_prob_test  = best.predict_proba(X_test)[:, 1]

    results["Tuned"] = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "AUC-ROC":  roc_auc_score(y_test, y_prob_test),
    }

    print("=== TRAINING ===")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_train, y_prob_train):.4f}")
    print("=== TEST ===")
    print(f"Accuracy: {results['Tuned']['Accuracy']:.4f}")
    print(f"AUC-ROC:  {results['Tuned']['AUC-ROC']:.4f}")
    print("=== CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(y_test, y_pred_test))

    # 2. Feature importance
    print(f"\n{'='*50}")
    print(f"{model_name} — Feature Importance")
    print(f"{'='*50}")

    importances = best.feature_importances_
    importance_df = pd.DataFrame({
        "feature":    X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(importance_df.to_string(index=False))

    plt.figure(figsize=(12, 8))
    sorted_idx = importances.argsort()[::-1]
    plt.bar(range(len(importances)), importances[sorted_idx])
    plt.xticks(range(len(importances)), X_train.columns[sorted_idx], rotation=90)
    plt.title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    plt.show()

    # 3. RFECV final model
    print(f"\n{'='*50}")
    print(f"{model_name} — Final model with selected features")
    print(f"{'='*50}")

    X_test_sel = X_test[selected_features]

    y_pred_final = final_model.predict(X_test_sel)
    y_prob_final = final_model.predict_proba(X_test_sel)[:, 1]

    results["RFECV"] = {
        "Accuracy": accuracy_score(y_test, y_pred_final),
        "AUC-ROC":  roc_auc_score(y_test, y_prob_final),
    }

    print(f"Selected features: {selected_features}")
    print(f"Final accuracy: {results['RFECV']['Accuracy']:.4f}")
    print(f"Final AUC-ROC:  {results['RFECV']['AUC-ROC']:.4f}")

    # 4. Without odds
    print(f"\n{'='*50}")
    print(f"{model_name} — Without bookmaker odds")
    print(f"{'='*50}")

    X_test_no_odds = X_test.drop(columns=ODDS_COLS)

    y_pred_no_odds = model_no_odds.predict(X_test_no_odds)
    y_prob_no_odds = model_no_odds.predict_proba(X_test_no_odds)[:, 1]

    results["No odds"] = {
        "Accuracy": accuracy_score(y_test, y_pred_no_odds),
        "AUC-ROC":  roc_auc_score(y_test, y_prob_no_odds),
    }

    print(f"Accuracy without odds: {results['No odds']['Accuracy']:.4f}")
    print(f"AUC-ROC without odds:  {results['No odds']['AUC-ROC']:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"{model_name} — Summary")
    print(f"{'='*50}")
    print(f"{'Variant':<12} {'Accuracy':>10} {'AUC-ROC':>10}")
    print("-" * 34)
    for variant, metrics in results.items():
        print(f"{variant:<12} {metrics['Accuracy']:>10.4f} {metrics['AUC-ROC']:>10.4f}")
        
        
        
def report_ann(model, X_train_scaled, X_test_scaled, y_train, y_test, model_name="ANN"):
    y_prob_train = model.predict(X_train_scaled, verbose=0).flatten()
    y_prob_test = model.predict(X_test_scaled,  verbose=0).flatten()
    y_pred_train = (y_prob_train > 0.5).astype(int)
    y_pred_test = (y_prob_test  > 0.5).astype(int)

    print(f"\n{'='*50}")
    print(f"{model_name} — Results")
    print(f"{'='*50}")
    print("=== TRAINING ===")
    print(f"Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_train, y_prob_train):.4f}")
    print("=== TEST ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_prob_test):.4f}")
    print("=== CLASSIFICATION REPORT (TEST) ===")
    print(classification_report(y_test, y_pred_test))




def plot_roc_curves(models_dict, y_test):
    
    plt.figure(figsize=(9, 7))
    for name, probs in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Model Comparison")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.show()




def report_error_analysis(y_test_vals, y_pred_arr, y_prob_arr, X_test_df, threshold=0.7):
    
    err_df = X_test_df.copy()
    err_df["y_true"] = y_test_vals
    err_df["y_pred"] = y_pred_arr
    err_df["y_prob"] = y_prob_arr
    err_df["confidence"] = np.abs(y_prob_arr - 0.5) * 2
    err_df["correct"] = (err_df["y_true"] == err_df["y_pred"]).astype(int)

    high_conf = err_df[(err_df["y_prob"] > threshold) | (err_df["y_prob"] < 1 - threshold)]
    high_conf_correct = high_conf[high_conf["correct"] == 1]
    high_conf_incorrect = high_conf[high_conf["correct"] == 0]

    print(f"High confidence (>{threshold}) correct:   {len(high_conf_correct)}")
    print(f"High confidence (>{threshold}) incorrect: {len(high_conf_incorrect)}")
    print()

    plt.figure(figsize=(6, 4))
    plt.hist(err_df[err_df["correct"] == 1]["confidence"], bins=30, alpha=0.7, label="Correct")
    plt.hist(err_df[err_df["correct"] == 0]["confidence"], bins=30, alpha=0.7, label="Incorrect")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Confidence distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Stratified accuracy by surface
    sub_df = X_test_df.copy()
    sub_df["y_true"] = y_test_vals
    sub_df["y_pred"] = y_pred_arr

    sub_df["surface"] = (
        sub_df[["Surface_Clay", "Surface_Grass", "Surface_Hard"]]
        .idxmax(axis=1)
        .str.replace("Surface_", "")
    )
    
    acc_surface = sub_df.groupby("surface").apply(
        lambda g: accuracy_score(g["y_true"], g["y_pred"])
    ).rename("Accuracy")
    
    print("\nAccuracy by surface:")
    print(acc_surface.round(4))

    # tournament stage
    sub_df["stage"] = pd.cut(
        sub_df["Round"],
        bins=[0, 2, 4, 10],
        labels=["Early rounds (1-2)", "Middle rounds (3-4)", "Late stages (5+)"],
    )
    
    acc_round = sub_df.groupby("stage", observed=True).apply(
        lambda g: accuracy_score(g["y_true"], g["y_pred"])
    ).rename("Accuracy")
    
    print("\nAccuracy by tournament stage:")
    print(acc_round.round(4))

    # ranking gap
    sub_df["gap_rank"] = pd.cut(
        sub_df["rank_diff"].abs(),
        bins=[0, 10, 50, 200, 10000],
        labels=["Small gap (0-10)", "Medium gap (10-50)", "Large gap (50-200)", "Huge gap (200+)"],
    )
    acc_rank = sub_df.groupby("gap_rank", observed=True).apply(
        lambda g: accuracy_score(g["y_true"], g["y_pred"])
    ).rename("Accuracy")
    print("\nAccuracy by ranking gap:")
    print(acc_rank.round(4))