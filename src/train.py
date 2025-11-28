import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_train_test(df: pd.DataFrame, test_size=0.3, random_state=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Unique pairs
    pairs = df[["country", "brand_name"]].drop_duplicates()

    train_pairs, eval_pairs = train_test_split(
        pairs,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    eval_pairs_s1, eval_pairs_s2 = train_test_split(
        eval_pairs,
        test_size=0.5,
        random_state=random_state,
        shuffle=True
    )

    # Build train and eval dataframes
    train_df = df.merge(train_pairs, on=["country", "brand_name"])
    eval_df_s1 = df.merge(eval_pairs_s1, on=["country", "brand_name"])
    eval_df_s2 = df.merge(eval_pairs_s2, on=["country", "brand_name"])

    return train_df, eval_df_s1, eval_df_s2


def split_true(df: pd.DataFrame, threshold: int, lags=3, rolling=5) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df_true = df[df['months_postgx'] >= threshold][['country', 'brand_name', 'months_postgx', 'volume']]
    df.loc[df['months_postgx'] >= threshold, 'volume'] = np.nan
    df.loc[df['months_postgx'] >= threshold, 'target_norm'] = np.nan
    for i in range(lags):
        df[f"lag{i+1}"] = np.nan
    df[f"roll{rolling}_mean"] = np.nan
    df[f"roll{rolling}_std"]  = np.nan
    df["pred"] = np.nan

    return df, df_true


def split_scenario(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_real = (
        df[df["volume"].notna()]
        .groupby(["country", "brand_name"])["months_postgx"]
        .max()
        .rename("max_real_month")
    )
    df_with_max = df.merge(max_real, on=["country", "brand_name"], how="left", validate="many_to_one")
    df_s1 = df_with_max[df_with_max["max_real_month"] == -1].copy().drop("max_real_month", axis=1)
    df_s2 = df_with_max[df_with_max["max_real_month"] == 5].copy().drop("max_real_month", axis=1)

    return df_s1, df_s2
