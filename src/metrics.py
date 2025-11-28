import numpy as np
import pandas as pd

def metric_s1(df: pd.DataFrame):
    """
    df must contain columns:
    - target_norm
    - pred
    - months_postgx
    - Avgj
    - country_brand_id  (integer id for grouping)
    """

    results = []

    for _, g in df.groupby(["country", "brand_name"]):
        avg = g["Avgj"].iloc[0]
        y_true = g["volume"].values
        y_pred = g["pred"].values * avg
        m = g["months_postgx"].values

        # Monthly error (0-23)
        mask_0_23 = (m >= 0) & (m <= 23)
        monthly_err = np.abs(y_true[mask_0_23] - y_pred[mask_0_23]).sum() / (24 * avg)

        # Accumulated error 0–5
        mask_0_5 = (m >= 0) & (m <= 5)
        acc_0_5 = np.abs(y_true[mask_0_5].sum() - y_pred[mask_0_5].sum()) / (6 * avg)

        # Accumulated error 6–11
        mask_6_11 = (m >= 6) & (m <= 11)
        acc_6_11 = np.abs(y_true[mask_6_11].sum() - y_pred[mask_6_11].sum()) / (6 * avg)

        # Accumulated error 12–23
        mask_12_23 = (m >= 12) & (m <= 23)
        acc_12_23 = np.abs(y_true[mask_12_23].sum() - y_pred[mask_12_23].sum()) / (12 * avg)

        # Weighted sum
        pe = (
            0.2 * monthly_err +
            0.5 * acc_0_5 +
            0.2 * acc_6_11 +
            0.1 * acc_12_23
        )

        results.append(pe)

    return np.mean(results)


def metric_s2(df: pd.DataFrame):
    results = []

    for _, g in df.groupby(["country", "brand_name"]):
        avg = g["Avgj"].iloc[0]
        y_true = g["volume"].values
        y_pred = g["pred"].values * avg
        m = g["months_postgx"].values

        # Monthly error (6-23)
        mask_6_23 = (m >= 6) & (m <= 23)
        monthly_err = np.abs(y_true[mask_6_23] - y_pred[mask_6_23]).sum() / (18 * avg)

        # Accumulated error 6–11
        mask_6_11 = (m >= 6) & (m <= 11)
        acc_6_11 = np.abs(y_true[mask_6_11].sum() - y_pred[mask_6_11].sum()) / (6 * avg)

        # Accumulated error 12–23
        mask_12_23 = (m >= 12) & (m <= 23)
        acc_12_23 = np.abs(y_true[mask_12_23].sum() - y_pred[mask_12_23].sum()) / (12 * avg)

        pe = (
            0.2 * monthly_err +
            0.5 * acc_6_11 +
            0.3 * acc_12_23
        )

        results.append(pe)

    return np.mean(results)
