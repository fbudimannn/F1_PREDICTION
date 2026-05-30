"""
ML Training Pipeline for F1 Qualifying Prediction.

Trains LightGBM Ranker and Quantile Regression models using historical data
(2022-2025) with Time-Series Cross-Validation, Optuna hyperparameter tuning,
and comprehensive anti-overfitting guards.

Usage:
    python train_model.py

Output:
    models/qualifying_ranker.joblib
    models/qualifying_ranker.txt
    models/lap_time_q10.joblib
    models/lap_time_q50.joblib
    models/lap_time_q90.joblib
    models/hyperparameters.json
    models/training_metrics.json
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODELS_DIR = "models"
DATA_PATH = "data/historical/training_features.csv"

# Feature columns used for training (all transferable across regulation eras)
FEATURE_COLS = [
    "driver_elo_prior",
    "constructor_pace_prior",
    "prev_quali_position",
    "avg_quali_position_last3",
]

TARGET_RANK = "target_quali_rank"
TARGET_TIME = "gap_to_pole"  # Use gap-to-pole as time target (transferable)


def load_data():
    """Load and validate training data. Returns None if data not found."""
    if not os.path.exists(DATA_PATH):
        print(f"Training data not found at {DATA_PATH}")
        print("Will generate synthetic training data as fallback.")
        return None

    df = pd.DataFrame(pd.read_csv(DATA_PATH))
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Features: {FEATURE_COLS}")

    # Drop rows with NaN in feature columns
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_RANK])
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing values")

    return df


def time_series_cv_split(df):
    """
    Time-Series Cross-Validation splits.
    Train on earlier years, validate on the next year.

    Returns list of (train_df, val_df, fold_name) tuples.
    """
    years = sorted(df['year'].unique())
    splits = []

    for i in range(1, len(years)):
        train_years = years[:i]
        val_year = years[i]
        train_df = df[df['year'].isin(train_years)]
        val_df = df[df['year'] == val_year]

        if len(train_df) >= 20 and len(val_df) >= 20:
            fold_name = f"Train {train_years} -> Val {val_year}"
            splits.append((train_df, val_df, fold_name))

    return splits


def compute_ndcg(y_true_ranks, y_pred_scores, k=5):
    """
    Compute NDCG@k for ranking evaluation.
    y_true_ranks: actual qualifying positions (1 = best)
    y_pred_scores: predicted relevance scores (higher = better predicted rank)
    """
    # Convert true ranks to relevance (higher rank = higher relevance)
    max_rank = max(y_true_ranks) if len(y_true_ranks) > 0 else 20
    relevance = [max(0, max_rank - r + 1) for r in y_true_ranks]

    # Sort by predicted scores (descending)
    combined = list(zip(y_pred_scores, relevance))
    combined.sort(key=lambda x: x[0], reverse=True)

    # DCG@k
    dcg = 0.0
    for i, (_, rel) in enumerate(combined[:k]):
        dcg += rel / np.log2(i + 2)

    # Ideal DCG@k
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance[:k]):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def train_ranker_with_optuna(df, n_trials=30):
    """
    Train LightGBM Ranker with Optuna hyperparameter optimization.
    Uses Time-Series CV for validation.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("Optuna not available, using default hyperparameters")
        return train_ranker_default(df)

    splits = time_series_cv_split(df)
    if len(splits) == 0:
        print("Not enough data for cross-validation, using default params")
        return train_ranker_default(df)

    best_params = None
    best_score = -1.0

    def objective(trial):
        nonlocal best_params, best_score

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 5, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        }

        cv_scores = []
        for train_df, val_df, _ in splits:
            X_train = train_df[FEATURE_COLS].values
            X_val = val_df[FEATURE_COLS].values

            # Relevance labels: invert rank so P1 = highest relevance
            max_rank_train = int(train_df[TARGET_RANK].max())
            y_train = (max_rank_train - train_df[TARGET_RANK].values + 1).astype(float)
            y_val_ranks = val_df[TARGET_RANK].values

            # Group sizes by race (year + round)
            train_groups = train_df.groupby(['year', 'round']).size().values.tolist()

            try:
                ranker = lgb.LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    ndcg_eval_at=[3, 5, 10],
                    verbose=-1,
                    **params
                )
                ranker.fit(X_train, y_train, group=train_groups)
                pred_scores = ranker.predict(X_val)
                ndcg = compute_ndcg(y_val_ranks.tolist(), pred_scores.tolist(), k=5)
                cv_scores.append(ndcg)
            except Exception:
                cv_scores.append(0.0)

        avg_score = float(np.mean(cv_scores)) if cv_scores else 0.0

        if avg_score > best_score:
            best_score = avg_score
            best_params = params.copy()

        return avg_score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"\nOptuna best NDCG@5: {best_score:.4f}")
    print(f"Best params: {best_params}")

    # Train final model with best params on all data except last year (for test)
    return best_params, best_score


def train_ranker_default(df):
    """Train with sensible default parameters when Optuna is unavailable."""
    params = {
        "n_estimators": 80,
        "learning_rate": 0.08,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 5,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
    }
    return params, 0.0


def train_and_save_models(df, best_params):
    """
    Train final models using best hyperparameters and save to disk.
    Train on 2022-2024, test on 2025 for final evaluation.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    years = sorted(df['year'].unique())
    if len(years) >= 2:
        test_year = years[-1]
        train_years = years[:-1]
    else:
        test_year = years[0]
        train_years = years

    train_df = df[df['year'].isin(train_years)]
    test_df = df[df['year'] == test_year]

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values

    max_rank_train = int(train_df[TARGET_RANK].max())
    y_train_rel = (max_rank_train - train_df[TARGET_RANK].values + 1).astype(float)
    y_test_ranks = test_df[TARGET_RANK].values

    train_groups = train_df.groupby(['year', 'round']).size().values.tolist()

    # ---- 1. Train LightGBM Ranker ----
    print("\n[1/2] Training LightGBM Ranker (LambdaMART)...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3, 5, 10],
        verbose=-1,
        **best_params
    )
    ranker.fit(X_train, y_train_rel, group=train_groups)

    # Evaluate on test set
    pred_scores = ranker.predict(X_test)
    test_ndcg3 = compute_ndcg(y_test_ranks.tolist(), pred_scores.tolist(), k=3)
    test_ndcg5 = compute_ndcg(y_test_ranks.tolist(), pred_scores.tolist(), k=5)
    test_ndcg10 = compute_ndcg(y_test_ranks.tolist(), pred_scores.tolist(), k=10)

    # Training NDCG for overfit check
    train_pred = ranker.predict(X_train)
    train_ndcg5 = compute_ndcg(
        train_df[TARGET_RANK].values.tolist(),
        train_pred.tolist(), k=5
    )
    overfit_gap = abs(train_ndcg5 - test_ndcg5)

    print(f"  Train NDCG@5: {train_ndcg5:.4f}")
    print(f"  Test NDCG@3:  {test_ndcg3:.4f}")
    print(f"  Test NDCG@5:  {test_ndcg5:.4f}")
    print(f"  Test NDCG@10: {test_ndcg10:.4f}")
    print(f"  Overfit gap:  {overfit_gap:.4f} {'(OK)' if overfit_gap < 0.15 else '(WARNING: possible overfit)'}")

    # Save ranker
    joblib.dump(ranker, os.path.join(MODELS_DIR, "qualifying_ranker.joblib"))
    ranker.booster_.save_model(os.path.join(MODELS_DIR, "qualifying_ranker.txt"))
    print(f"  Saved: qualifying_ranker.joblib + qualifying_ranker.txt")

    # ---- 2. Train Quantile Regression Models ----
    print("\n[2/2] Training Quantile Regression models (gap-to-pole)...")
    y_train_time = train_df[TARGET_TIME].values
    y_test_time = test_df[TARGET_TIME].values

    quantile_metrics = {}
    for q, label in [(0.10, "best_case"), (0.50, "median"), (0.90, "worst_case")]:
        q_params = best_params.copy()
        q_model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            verbose=-1,
            **q_params
        )
        q_model.fit(X_train, y_train_time)
        q_pred = q_model.predict(X_test)
        mae = float(np.mean(np.abs(q_pred - y_test_time)))

        fname = f"lap_time_q{int(q*100):02d}.joblib"
        joblib.dump(q_model, os.path.join(MODELS_DIR, fname))
        print(f"  Q{int(q*100)}% ({label}): MAE = {mae:.3f}s | Saved: {fname}")
        quantile_metrics[f"q{int(q*100)}_mae"] = mae

    # ---- 3. Save metrics and hyperparameters ----
    metrics = {
        "train_years": [int(y) for y in train_years],
        "test_year": int(test_year),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_ndcg5": round(train_ndcg5, 4),
        "test_ndcg3": round(test_ndcg3, 4),
        "test_ndcg5": round(test_ndcg5, 4),
        "test_ndcg10": round(test_ndcg10, 4),
        "overfit_gap": round(overfit_gap, 4),
        "overfit_status": "OK" if overfit_gap < 0.15 else "WARNING",
        **quantile_metrics
    }

    with open(os.path.join(MODELS_DIR, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODELS_DIR, "hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n  Saved: training_metrics.json + hyperparameters.json")

    # ---- 4. Feature Importance ----
    print("\n  Feature Importance (Ranker):")
    importances = ranker.feature_importances_
    for feat, imp in sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True):
        print(f"    {feat}: {imp}")

    return metrics


def main():
    print("=" * 60)
    print("F1 ML Training Pipeline")
    print("LightGBM Ranker + Quantile Regression")
    print("=" * 60)

    df = load_data()

    if df is None or len(df) < 50:
        if df is not None:
            print(f"\nInsufficient data ({len(df)} rows). Need at least 50 rows.")
        print("Generating synthetic training data as fallback...")
        df = generate_synthetic_fallback()

    # Hyperparameter tuning with Optuna
    print("\n--- Hyperparameter Tuning (Optuna) ---")
    best_params, optuna_score = train_ranker_with_optuna(df, n_trials=30)

    # Train final models and evaluate
    print("\n--- Final Model Training & Evaluation ---")
    metrics = train_and_save_models(df, best_params)

    print("\n" + "=" * 60)
    print("Training pipeline complete!")
    print(f"Models saved to: {MODELS_DIR}/")
    print(f"Test NDCG@5: {metrics['test_ndcg5']:.4f}")
    print(f"Overfit status: {metrics['overfit_status']}")
    print("=" * 60)


def generate_synthetic_fallback():
    """
    Generate synthetic training data when historical data is insufficient.
    This ensures the training pipeline can still produce usable models.
    """
    print("Generating synthetic training data for model calibration...")
    np.random.seed(42)
    rows = []

    for year in [2022, 2023, 2024, 2025]:
        for rnd in range(1, 24):
            n_drivers = 20
            elos = np.random.normal(1600, 150, n_drivers)
            constructor_paces = np.random.uniform(-0.6, 0.5, n_drivers)

            # Simulate qualifying: lower time = better position
            true_pace = -elos * 0.001 + constructor_paces + np.random.normal(0, 0.1, n_drivers)
            ranks = np.argsort(np.argsort(true_pace)) + 1
            gap_to_pole = true_pace - true_pace.min()

            for i in range(n_drivers):
                tm_gap = np.random.normal(0, 0.3)
                rows.append({
                    'year': year,
                    'round': rnd,
                    'driver': f"DR{i:02d}",
                    'team': f"Team{i // 2}",
                    'driver_elo_raw_current': elos[i],
                    'teammate_quali_gap': tm_gap,
                    'gap_to_pole': gap_to_pole[i],
                    'constructor_pace_raw_current': constructor_paces[i],
                    'quali_position': int(ranks[i]),
                    'grid_position': int(ranks[i]),
                    'finish_position': int(np.clip(ranks[i] + np.random.randint(-3, 4), 1, 20)),
                    'is_dnf': 0,
                    'teammate_race_gap': np.random.normal(0, 2),
                    'target_quali_rank': int(ranks[i]),
                    'target_race_rank': int(np.clip(ranks[i] + np.random.randint(-3, 4), 1, 20)),
                })

    df = pd.DataFrame(rows)
    
    # Compute driver_elo_prior
    driver_elos = {}
    elo_column = []
    for _, row in df.sort_values(['year', 'round']).iterrows():
        driver = row['driver']
        if driver not in driver_elos:
            driver_elos[driver] = 1500.0
        elo_column.append(driver_elos[driver])
        
        pos = row['quali_position']
        expected_pos = 10.0
        delta = (expected_pos - pos) * 1.5
        driver_elos[driver] = driver_elos[driver] + delta
        
    df = df.sort_values(['year', 'round']).reset_index(drop=True)
    df['driver_elo_prior'] = elo_column

    # Compute constructor_pace_prior
    team_gaps = df.groupby(['year', 'round', 'team'])['gap_to_pole'].min().reset_index()
    team_gaps = team_gaps.sort_values(['team', 'year', 'round'])
    team_gaps['constructor_pace_rolling'] = team_gaps.groupby('team')['gap_to_pole'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    team_gaps['constructor_pace_prior'] = team_gaps.groupby('team')['constructor_pace_rolling'].shift(1)
    
    df = pd.merge(
        df,
        team_gaps[['year', 'round', 'team', 'constructor_pace_prior']],
        on=['year', 'round', 'team'],
        how='left'
    )
    df['constructor_pace_prior'] = df['constructor_pace_prior'].fillna(0.5)

    # Compute prev_quali_position and avg_quali_position_last3
    df['prev_quali_position'] = df.groupby('driver')['quali_position'].shift(1).fillna(11.0)
    df['avg_quali_position_last3'] = df.groupby('driver')['quali_position'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(11.0)

    # Clean up temporary columns
    df = df.drop(columns=['driver_elo_raw_current', 'constructor_pace_raw_current'])

    # Save synthetic data
    os.makedirs("data/historical", exist_ok=True)
    df.to_csv("data/historical/training_features.csv", index=False)
    print(f"Synthetic data: {len(df)} rows generated and saved.")
    return df


if __name__ == "__main__":
    main()
