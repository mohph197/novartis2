from catboost import CatBoostRegressor, Pool, cv
import pandas as pd
import numpy as np
from typing import Union, Optional
from .consts import LAGS, ROLLING

cat_features = ['month', 'ther_area', 'main_package', 'biological', 'small_molecule']

num_features = [
    'n_gxs', 'hospital_rate', 'Avgj',
    'pre_mean', 'pre_std', 'pre_min', 'pre_max', 'pre_trend',
    't1_mean', 't1_std', 't1_min', 't1_max', 't1_trend',
    't2_mean', 't2_std', 't2_min', 't2_max', 't2_trend',
    't3_mean', 't3_std', 't3_min', 't3_max', 't3_trend',
    't4_mean', 't4_std', 't4_min', 't4_max', 't4_trend',
    't5_mean', 't5_std', 't5_min', 't5_max', 't5_trend',
    't6_mean', 't6_std', 't6_min', 't6_max', 't6_trend',
    't7_mean', 't7_std', 't7_min', 't7_max', 't7_trend',
    't8_mean', 't8_std', 't8_min', 't8_max', 't8_trend',
    # "ratio_last3_pre12", "ratio_last6_pre12",
    # "slope_change_6_12", "pre_cv",
    'month_sin', 'month_cos', 'months_postgx',
]

for i in range(1, LAGS+1):
    num_features.append(f"lag{i}")
    # num_features.append(f"ngxs_lag{i}")

num_features.append(f"roll{ROLLING}_mean")
num_features.append(f"roll{ROLLING}_std")
# num_features.append(f"ngxs_roll{ROLLING}_mean")
# num_features.append(f"ngxs_roll{ROLLING}_std")


features = cat_features + num_features


def fit_model(df: pd.DataFrame, scenario: str, iters=2000, lr=0.03, seed=42, verbose=True, predict_avg=False) -> CatBoostRegressor:
    threshold = 0 if scenario == "s1" else 6
    train_pool = Pool(
        data=df[df['months_postgx'] >= threshold][features],
        label=df[df['months_postgx'] >= threshold]['target_norm_avg' if predict_avg else 'target_norm'],
        cat_features=[features.index(c) for c in cat_features],
        feature_names=features,
        weight=df[df['months_postgx'] >= threshold][f'weight_{scenario}']
    )

    model = CatBoostRegressor(
        iterations=iters,
        learning_rate=lr,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200 if verbose else 0
    )

    return model.fit(train_pool)


def fit_model_cv(df: pd.DataFrame, scenario: str, iters=2000, lr=0.03, seed=42, verbose=True, predict_avg=False, folds=3) -> CatBoostRegressor:
    # 1. Prepare Data (Identical to your original function)
    threshold = 0 if scenario == "s1" else 6

    # Create the Pool exactly as you did before
    train_pool = Pool(
        data=df[df['months_postgx'] >= threshold][features],
        label=df[df['months_postgx'] >= threshold]['target_norm_avg' if predict_avg else 'target_norm'],
        cat_features=[features.index(c) for c in cat_features]
    )
    # Apply weights to the pool so CV respects them
    train_pool.set_weight(df[df['months_postgx'] >= threshold][f'weight_{scenario}'])

    # 2. Define Parameters for CV
    # Note: We move params into a dict to pass to the cv() function
    params = {
        'iterations': iters,
        'learning_rate': lr,
        'depth': 8,
        'loss_function': "MAE",
        'eval_metric': "MAE",
        'random_seed': seed,
        'verbose': 0  # Keep CV silent to avoid spamming 3x logs, we will print the result manually
    }

    # 3. Run Cross-Validation
    if verbose:
        print(f"Running {folds}-fold CV to find optimal iterations...")

    cv_data: pd.DataFrame = cv(
        pool=train_pool,
        params=params,
        fold_count=folds,          # Split data into 3 (or more) parts
        early_stopping_rounds=50,  # Stop if validation MAE doesn't improve for 50 rounds
        shuffle=True,              # Important: shuffle data before splitting
        seed=seed,
        logging_level='Verbose' if verbose else 'Silent'
    )

    # 4. Extract the Best Iteration
    # cv_data is a DataFrame with columns like 'test-MAE-mean' and 'train-MAE-mean'
    best_iteration = cv_data['test-MAE-mean'].idxmin()
    best_mae = cv_data['test-MAE-mean'].min()

    if verbose:
        print(f"  > Optimal Iterations Found: {best_iteration}")
        print(f"  > Best CV MAE: {best_mae:.4f}")

    # 5. Retrain Final Model
    # We initialize a NEW model using the discovered best_iteration
    final_model = CatBoostRegressor(
        iterations=best_iteration, # <--- The auto-tuned value
        learning_rate=lr,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200 if verbose else 0
    )

    # Fit on the FULL dataset (train_pool)
    return final_model.fit(train_pool)


def predict(model: Union[CatBoostRegressor, list[CatBoostRegressor]], df: pd.DataFrame, threshold: int, votes: Optional[pd.DataFrame]=None, lags=LAGS, rolling=ROLLING):
    preds = []
    buffer_size = max(rolling, lags)
    groups = df.groupby(["country", "brand_name"])

    for (country, brand), g in groups:

        g = g.sort_values("months_postgx")

        history = g[g['months_postgx'].isin(range(threshold - buffer_size, threshold))]["target_norm"].tolist()

        for idx, row in g.iterrows():
            if row['months_postgx'] < threshold:
                continue

            # ----- 1. Insert lag features -----
            for i in range(lags):
                g.loc[idx, f"lag{i+1}"] = history[-(i+1)]

            # ----- 2. Compute rolling features -----
            g.loc[idx, f"roll{rolling}_mean"] = np.mean(history[-rolling:])
            g.loc[idx, f"roll{rolling}_std"]  = np.std(history[-rolling:])

            # ----- 3. Prepare row for prediction -----
            X_row = g.loc[idx, features]

            # Predict
            if isinstance(model, list):
                votes_row: pd.DataFrame = votes[(votes["country"] == country) & (votes["brand_name"] == brand) & (votes["months_postgx"] == row["months_postgx"])]
                assert len(votes_row) == 1
                votes_row = votes_row.iloc[:, 3:].values.flatten()
                if len(votes_row) != len(model):
                    raise ValueError(f"Votes shape mismatch: {len(votes_row)} != {len(model)}")
                pred = (votes_row * np.array([m.predict(X_row.values.reshape(1, -1))[0] for m in model])).sum()
            else:
                pred = model.predict(X_row.values.reshape(1, -1))[0]

            # Save prediction
            history.append(pred)
            assert np.isnan(g.loc[idx, "target_norm"]) and np.isnan(g.loc[idx, "volume"])
            g.loc[idx, "target_norm"] = pred
            g.loc[idx, "volume"] = pred * g.loc[idx, "Avgj"]

        preds.append(g)

    # Combine predictions
    return pd.concat(preds).sort_index()
