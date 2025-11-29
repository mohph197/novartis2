import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from .preprocess import build_bucket_dataset

cat_features = ['ther_area', 'main_package', 'biological', 'small_molecule']  # TODO: Test including country

def num_features(scenario: str):
    f = [
        'hospital_rate', 'Avgj',
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
        "n_gxs_0", "n_gxs_6", "n_gxs_12", "n_gxs_23", "n_gxs_trend"
    ]

    if scenario == "s2":
        f += [
            "post0_5_mean","post0_5_min","post0_5_max","post0_5_std",
            "post0_5_trend","post_drop0_1"
        ]

    return f


def fit_bucket_classifier(base: pd.DataFrame, scenario: str, iters=2000, lr=0.03, seed=42, verbose=True) -> CatBoostClassifier:
    features = cat_features + num_features(scenario)

    X = base[features]
    y = base["label"].values

    # class weights: upweight high-erosion (label=1)
    class_weights = {0: 1.0, 1: 20.0}

    pool = Pool(
        data=X,
        label=y,
        cat_features=[features.index(c) for c in cat_features],
        feature_names=features,
        weight=np.where(y == 1, class_weights[1], class_weights[0])
    )

    model = CatBoostClassifier(
        iterations=iters,
        learning_rate=lr,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=200 if verbose else 0
    )

    return model.fit(pool)


def predict_bucket1_proba(model: CatBoostClassifier, base: pd.DataFrame, scenario: str) -> np.ndarray:
    features = cat_features + num_features(scenario)
    return model.predict_proba(base[features])[:, 1]


def predict_probas(model: CatBoostClassifier, df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    base = build_bucket_dataset(df, None, scenario)

    features = cat_features + num_features(scenario)
    probs = model.predict_proba(base[features])

    df_probs = base[["country", "brand_name"]].copy()
    df_probs["prob_bucket1"] = probs[:, 1]
    df_probs["prob_bucket2"] = probs[:, 0]

    return df[["country", "brand_name", "months_postgx"]].merge(df_probs, on=["country", "brand_name"], how="left", validate="many_to_one")
