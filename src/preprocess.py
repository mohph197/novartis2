import pandas as pd
import numpy as np
from .consts import LAGS, ROLLING
from typing import Union, Optional


month_to_int = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}


def merge_dfs(vol: pd.DataFrame, gxs: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    df = vol.merge(
        gxs,
        on=['country', 'brand_name', 'months_postgx'],
        how='left',
        validate='one_to_one'
    )
    df = df.merge(
        info,
        on=['country', 'brand_name'],
        how='left',
        validate='many_to_one'
    )

    return df


def compute_avg_and_normalize(df: pd.DataFrame) -> pd.DataFrame:
    avgj = (
        df[df["months_postgx"].between(-12, -1)]
        .groupby(["country", "brand_name"])["volume"]
        .mean()
        .rename("Avgj")
    )

    df = df.merge(avgj, on=["country", "brand_name"], how="left", validate="many_to_one")

    df["target_norm"] = df["volume"] / df["Avgj"]

    return df


def create_mean_targets(df: pd.DataFrame, rolling_mean: bool, calibrate: bool) -> pd.DataFrame:
    df = df.copy()

    if rolling_mean:
        df['target_norm_avg'] = df.groupby(['country', 'brand_name'])['target_norm'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    else:
        df['interval_group'] = pd.cut(df['months_postgx'], bins=[-1, 5, 11, 23])
        df['target_norm_avg'] = df.groupby(['country', 'brand_name', 'interval_group'], observed=True)['target_norm'].transform('mean')
        df = df.drop(columns=['interval_group'])

    if calibrate:
        df['target_norm_avg'] = 0.8 * df['target_norm_avg'] + 0.2 * df['target_norm']

    return df


def pre_features(df: pd.DataFrame, normalize: bool) -> pd.DataFrame:
    pre = df[df["months_postgx"] < 0]

    target_col = "target_norm" if normalize else "volume"

    # --- global pre stats ---
    pre_stats = pre.groupby(["country", "brand_name"])[target_col].agg(
        pre_mean="mean",
        pre_std="std",
        pre_min="min",
        pre_max="max",
        pre_trend=lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
    )

    df = df.merge(pre_stats, on=["country", "brand_name"], how="left", validate="many_to_one")

    # # helper fns operating on a Series ordered by months_postgx
    # def _lastk_mean(x, k):
    #     # x = x.dropna()
    #     assert not x.isna().any()
    #     return x.tail(k).mean() if len(x) else 0.0

    # def _slope_tail(x, k):
    #     x = x.tail(k)
    #     # x = x.dropna()
    #     assert not x.isna().any()
    #     if len(x) <= 1:
    #         return 0.0
    #     return np.polyfit(np.arange(len(x)), x, 1)[0]

    # pre_shape = pre.sort_values("months_postgx").groupby(["country", "brand_name"])[target_col].agg(
    #     pre12_mean_tmp=lambda x: _lastk_mean(x, 12),
    #     last3_mean_tmp=lambda x: _lastk_mean(x, 3),
    #     last6_mean_tmp=lambda x: _lastk_mean(x, 6),
    #     slope_last6_tmp=lambda x: _slope_tail(x, 6),
    #     slope_last12_tmp=lambda x: _slope_tail(x, 12),
    # )


    # # ratios + slope change + volatility
    # eps = 1e-9
    # pre_shape["ratio_last3_pre12"] = pre_shape["last3_mean_tmp"] / (pre_shape["pre12_mean_tmp"] + eps)
    # pre_shape["ratio_last6_pre12"] = pre_shape["last6_mean_tmp"] / (pre_shape["pre12_mean_tmp"] + eps)
    # pre_shape["slope_change_6_12"] = pre_shape["slope_last6_tmp"] - pre_shape["slope_last12_tmp"]

    # # coefficient of variation (robust volatility proxy)
    # pre_shape["pre_cv"] = pre_stats["pre_std"] / (pre_stats["pre_mean"] + eps)

    # # keep only engineered cols
    # pre_shape = pre_shape[[
    #     "ratio_last3_pre12",
    #     "ratio_last6_pre12",
    #     "slope_change_6_12",
    #     "pre_cv",
    # ]]

    # df = df.merge(pre_shape, on=["country", "brand_name"], how="left", validate="many_to_one")

    windows = {
        "t1": (-24, -22),
        "t2": (-21, -19),
        "t3": (-18, -16),
        "t4": (-15, -13),
        "t5": (-12, -10),
        "t6": (-9, -7),
        "t7": (-6, -4),
        "t8": (-3, -1),
    }

    trimester_frames = []

    for name, (start, end) in windows.items():
        tmp = (
            pre[pre["months_postgx"].between(start, end)]
            .groupby(["country", "brand_name"])[target_col]
            .agg(
                **{
                    f"{name}_mean": "mean",
                    f"{name}_std": "std",
                    f"{name}_min": "min",
                    f"{name}_max": "max",
                    f"{name}_trend": lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                }
            )
        )
        trimester_frames.append(tmp)

    trimester_stats = pd.concat(trimester_frames, axis=1)

    return df.merge(trimester_stats, on=["country", "brand_name"], how="left", validate="many_to_one")


def month_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['month_sin'] = np.sin(df['month'] * (2.*np.pi/12))
    df['month_cos'] = np.cos(df['month'] * (2.*np.pi/12))

    return df


def n_gxs_features(df: pd.DataFrame, lags=LAGS, rolling=ROLLING) -> pd.DataFrame:
    df = df.copy()

    for i in range(lags):
        df[f"ngxs_lag{i+1}"] = df.groupby(["country", "brand_name"])["n_gxs"].shift(i + 1)
    df['ngxs_roll{r}_mean'.format(r=rolling)] = df.groupby(["country", "brand_name"])["n_gxs"].rolling(rolling).mean().reset_index()['n_gxs']
    df['ngxs_roll{r}_std'.format(r=rolling)] = df.groupby(["country", "brand_name"])["n_gxs"].rolling(rolling).mean().reset_index()['n_gxs']

    return df


def build_bucket_dataset(df: pd.DataFrame, df_mge: pd.DataFrame, df_aux: Optional[pd.DataFrame], scenario: str) -> pd.DataFrame:
    base = (
        df.sort_values("months_postgx")
          .groupby(["country", "brand_name"], as_index=False)
          .first()
    )

    # ----- generic snapshots (less noisy than rolling) -----
    for m in [0, 6, 12, 23]:
        snap = df.loc[df["months_postgx"] == m, ["country", "brand_name", "n_gxs"]]
        snap = snap.rename(columns={"n_gxs": f"n_gxs_{m}"})
        base = base.merge(snap, on=["country", "brand_name"], how="left")
    base['n_gxs_trend'] = (df[df["months_postgx"] >= 0]
        .groupby(["country", "brand_name"])["n_gxs"]
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)>1 else 0)
        .reset_index()['n_gxs']
    )

    # ----- scenario 2 extra post-0..5 early erosion features -----
    if scenario == "s2":
        post = df[df["months_postgx"].between(0, 5)].copy()

        post_feats = (
            post.groupby(["country", "brand_name"])["target_norm"]
                .agg(
                    post0_5_mean="mean",
                    post0_5_min="min",
                    post0_5_max="max",
                    post0_5_std="std",
                    post0_5_trend=lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)>1 else 0
                )
                .reset_index()
        )

        # early drop 0->1 if available
        post01 = (
            post[post["months_postgx"].isin([0, 1])]
            .pivot_table(index=["country", "brand_name"],
                         columns="months_postgx",
                         values="target_norm")
            .reset_index()
        )
        if 0 in post01.columns and 1 in post01.columns:
            post01["post_drop0_1"] = post01[1] - post01[0]
        else:
            post01["post_drop0_1"] = 0.0
        post01 = post01[["country", "brand_name", "post_drop0_1"]]

        base = base.merge(post_feats, on=["country", "brand_name"], how="left")
        base = base.merge(post01, on=["country", "brand_name"], how="left")
        # base[["post0_5_mean","post0_5_min","post0_5_max","post0_5_std","post0_5_trend","post_drop0_1"]] = \
        #     base[["post0_5_mean","post0_5_min","post0_5_max","post0_5_std","post0_5_trend","post_drop0_1"]].fillna(0)
        if base[["post0_5_mean","post0_5_min","post0_5_max","post0_5_std","post0_5_trend","post_drop0_1"]].isna().any().any():
            raise ValueError("NAs in post-0..5 early erosion features")

    # ----- attach mge stats -----
    base = base.merge(df_mge[[
        'country', 'brand_name',
        'country_mge_min', 'country_mge_max', 'country_mge_mean', 'country_mge_std',
        'brand_mge_min', 'brand_mge_max', 'brand_mge_mean', 'brand_mge_std',
    ]], on=['country', 'brand_name'], how='left', validate='one_to_one')

    # Fill NaN mge stats with stats of global MGE
    base['country_mge_min'] = base['country_mge_min'].fillna(df_mge['MGE'].min())
    base['country_mge_max'] = base['country_mge_max'].fillna(df_mge['MGE'].max())
    base['country_mge_mean'] = base['country_mge_mean'].fillna(df_mge['MGE'].mean())
    base['country_mge_std'] = base['country_mge_std'].fillna(df_mge['MGE'].std())
    base['brand_mge_min'] = base['brand_mge_min'].fillna(df_mge['MGE'].min())
    base['brand_mge_max'] = base['brand_mge_max'].fillna(df_mge['MGE'].max())
    base['brand_mge_mean'] = base['brand_mge_mean'].fillna(df_mge['MGE'].mean())
    base['brand_mge_std'] = base['brand_mge_std'].fillna(df_mge['MGE'].std())

    if df_aux is None:
        return base

    # ----- attach bucket label -----
    y = df_aux[["country", "brand_name", "bucket"]].copy()
    base = base.merge(y, on=["country", "brand_name"], how="inner", validate="one_to_one")

    # map bucket {1,2} -> label {1,0} (positive = high erosion)
    base["label"] = (base["bucket"] == 1).astype(int)
    base = base.drop(columns=["bucket"])

    return base


def create_bucket_labels(df: pd.DataFrame, df_aux: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.sort_values("months_postgx")
          .groupby(["country", "brand_name"], as_index=False)
          .first()
          [["country", "brand_name"]]
          .merge(df_aux[["country", "brand_name", "bucket"]], on=["country", "brand_name"], how="left", validate="one_to_one")
    )

    base["label"] = (base["bucket"] == 1).astype(int)
    base = base.drop(columns=["bucket"])

    return base


def general_preprocessing(
    vol_df: pd.DataFrame,
    gxs_df: pd.DataFrame,
    info_df: pd.DataFrame,
    is_test=False,
    rolling_mean=False,
    normalize_pre=True,
    calibrate_target_avg=False,
) -> tuple[pd.DataFrame, ...]:
    vol_df['month'] = vol_df["month"].map(month_to_int)
    if is_test:
        vol_df, sub_df = extend_test(vol_df)
    df = merge_dfs(vol_df, gxs_df, info_df)
    df = compute_avg_and_normalize(df)
    df = create_mean_targets(df, rolling_mean, calibrate_target_avg)
    df = pre_features(df, normalize=normalize_pre)
    df = month_features(df)
    # df = n_gxs_features(df)

    if is_test:
        return df, sub_df

    df_aux = create_aux(df)
    return df, df_aux


def add_lags(df: pd.DataFrame, lags=LAGS, rolling=ROLLING) -> pd.DataFrame:
    df = df.copy()
    for i in range(lags):
        df[f"lag{i+1}"] = df.groupby(["country", "brand_name"])["target_norm"].shift(i + 1)

    df['roll{r}_mean'.format(r=rolling)] = df.groupby(["country", "brand_name"])["target_norm"].rolling(rolling).mean().reset_index()['target_norm']
    df['roll{r}_std'.format(r=rolling)] = df.groupby(["country", "brand_name"])["target_norm"].rolling(rolling).mean().reset_index()['target_norm']

    return df


def add_weights(df: pd.DataFrame, df_aux: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['weight_s1'] = np.nan
    df['weight_s2'] = np.nan

    df.loc[(df['months_postgx'] >= 0) & (df['months_postgx'] < 6), 'weight_s1'] = 0.5 / 0.8
    df.loc[(df['months_postgx'] >= 6) & (df['months_postgx'] < 12), 'weight_s1'] = 0.2 / 0.8
    df.loc[(df['months_postgx'] >= 12) & (df['months_postgx'] < 24), 'weight_s1'] = 0.1 / 0.8

    df.loc[(df['months_postgx'] >= 6) & (df['months_postgx'] < 12), 'weight_s2'] = 0.5 / 0.8
    df.loc[(df['months_postgx'] >= 12) & (df['months_postgx'] < 24), 'weight_s2'] = 0.3 / 0.8

    df = df.merge(df_aux[['country', 'brand_name', 'bucket']], on=['country', 'brand_name'], how='left', validate='many_to_one')
    df['bucket_weight'] = df.apply(lambda x: 2 if x['bucket'] == 1 else 1, axis=1)
    df["weight_s1"] = df['weight_s1'] * df['bucket_weight']
    df["weight_s2"] = df['weight_s2'] * df['bucket_weight']
    df = df.drop(columns=['bucket', 'bucket_weight'])

    return df


def train_preprocessing(df: pd.DataFrame, df_aux: pd.DataFrame) -> pd.DataFrame:
    df = add_lags(df)
    df = add_weights(df, df_aux)

    return df


def extend_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    extended_rows = []
    submission_rows = []

    for (country, brand), g in df.groupby(["country", "brand_name"]):

        g = g.sort_values("months_postgx")

        # last pre-gx row (months_postgx = -1)
        last_row = g.iloc[-1]
        last_moth_postgx = last_row["months_postgx"]
        last_month = last_row["month"]

        # create new rows for horizon months_postgx = 0..23
        for i, h in enumerate(list(range(last_moth_postgx + 1, 24))):

            # wrap month: 1..12
            new_month = ((last_month + i) % 12) + 1

            extended_rows.append({
                "country": country,
                "brand_name": brand,
                "month": new_month,
                "months_postgx": h,
                "volume": np.nan
            })

            submission_rows.append({
                "country": country,
                "brand_name": brand,
                "months_postgx": h
            })

    # Build future DF
    df_extension = pd.DataFrame(extended_rows)
    df_submission = pd.DataFrame(submission_rows)

    # Combine original + extended rows
    df = pd.concat([df, df_extension], ignore_index=True)

    df = df.sort_values(
        ["country", "brand_name", "months_postgx"]
    ).reset_index(drop=True)

    return df, df_submission


def create_aux(df: pd.DataFrame) -> pd.DataFrame:
    df_aux = df[df['months_postgx'] >= 0].copy()
    df_aux['volume_norm'] = df_aux['volume'] / df_aux['Avgj']
    df_aux = df_aux.groupby(['country','brand_name'])['volume_norm'].mean().rename('MGE').reset_index()
    df_aux['bucket'] = df_aux['MGE'].apply(lambda x: 1 if x <= 0.25 else 2)
    # df_aux = df_aux.drop(columns=['MGE'])
    df_aux["avg_vol"] = df[df['months_postgx'] >= 0].groupby(["country", "brand_name"])["Avgj"].first().reset_index()["Avgj"]

    return df_aux


def build_mge_dataset(df: pd.DataFrame, df_aux: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.sort_values("months_postgx")
          .groupby(["country", "brand_name"], as_index=False)
          .first()
          [["country", "brand_name"]]
    )

    base = base.merge(df_aux[['country', 'brand_name', 'MGE']], on=['country', 'brand_name'], how='left', validate='one_to_one')

    country_mge_stats = base.groupby('country')['MGE'].agg(
        country_mge_min="min",
        country_mge_max="max",
        country_mge_mean="mean",
        country_mge_std="std",
    )

    base = base.merge(country_mge_stats, on='country', how='left', validate='many_to_one')

    brand_mge_stats = base.groupby('brand_name')['MGE'].agg(
        brand_mge_min="min",
        brand_mge_max="max",
        brand_mge_mean="mean",
        brand_mge_std="std",
    )

    base = base.merge(brand_mge_stats, on='brand_name', how='left', validate='many_to_one')

    base['country_mge_std'] = base['country_mge_std'].fillna(0)
    base['brand_mge_std'] = base['brand_mge_std'].fillna(0)

    assert not base.isna().any().any(), "NaNs found!"

    return base
