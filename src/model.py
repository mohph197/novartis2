from catboost import CatBoostRegressor, Pool
import pandas as pd
import numpy as np
from .consts import LAGS, ROLLING

cat_features = [
    'month', 'ther_area', 'main_package',
    'biological', 'small_molecule'
]

num_features = [
    'n_gxs', 'hospital_rate',
    'Avgj', 'pre_mean', 'pre_std',
    'pre_min', 'pre_max', 'pre_trend', 't1_mean', 't1_std', 't1_min',
    't1_max', 't1_trend', 't2_mean', 't2_std', 't2_min', 't2_max',
    't2_trend', 't3_mean', 't3_std', 't3_min', 't3_max', 't3_trend',
    't4_mean', 't4_std', 't4_min', 't4_max', 't4_trend', 't5_mean',
    't5_std', 't5_min', 't5_max', 't5_trend', 't6_mean', 't6_std', 't6_min',
    't6_max', 't6_trend', 't7_mean', 't7_std', 't7_min', 't7_max',
    't7_trend', 't8_mean', 't8_std', 't8_min', 't8_max', 't8_trend',
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


def fit_model(df: pd.DataFrame, threshold: int, iters=2000, seed=None, verbose=True) -> CatBoostRegressor:
    train_pool = Pool(
        data=df[df['months_postgx'] >= threshold][features],
        label=df[df['months_postgx'] >= threshold]['target_norm'],
        cat_features=[features.index(c) for c in cat_features]
    )

    model = CatBoostRegressor(
        iterations=iters,
        learning_rate=0.03,
        depth=8,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200 if verbose else 0
    )

    return model.fit(train_pool)


def predict(model: CatBoostRegressor, df: pd.DataFrame, threshold: int, lags=LAGS, rolling=ROLLING):
    preds = []
    buffer_size = max(rolling, lags)
    groups = df.groupby(["country", "brand_name"])

    for _, g in groups:

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
            pred = model.predict(X_row.values.reshape(1, -1))[0]

            # Save prediction
            history.append(pred)
            assert np.isnan(g.loc[idx, "target_norm"]) and np.isnan(g.loc[idx, "volume"])
            g.loc[idx, "target_norm"] = pred
            g.loc[idx, "volume"] = pred * g.loc[idx, "Avgj"]

        preds.append(g)

    # Combine predictions
    return pd.concat(preds).sort_index()
