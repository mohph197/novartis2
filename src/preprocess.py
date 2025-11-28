import pandas as pd
import numpy as np


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


def pre_features(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df["months_postgx"] < 0]

    pre_stats = pre.groupby(["country", "brand_name"])["target_norm"].agg(
        pre_mean="mean",
        pre_std="std",
        pre_min="min",
        pre_max="max",
        pre_trend=lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)>1 else 0
    )

    df = df.merge(pre_stats, on=["country", "brand_name"], how="left", validate="many_to_one")

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
            .groupby(["country", "brand_name"])["target_norm"]
            .agg(
                **{
                    f"{name}_mean": "mean",
                    f"{name}_std": "std",
                    f"{name}_min": "min",
                    f"{name}_max": "max",
                    f"{name}_trend": lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)>1 else 0,
                }
            )
        )
        trimester_frames.append(tmp)

    # Combine trimester stats
    trimester_stats = pd.concat(trimester_frames, axis=1)

    return df.merge(trimester_stats, on=["country", "brand_name"], how="left", validate="many_to_one")


def month_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['months_postgx_sin'] = np.sin(df['months_postgx'] * (2.*np.pi/12))
    df['months_postgx_cos'] = np.cos(df['months_postgx'] * (2.*np.pi/12))

    return df


def general_preprocessing(vol_df: pd.DataFrame, gxs_df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    df = merge_dfs(vol_df, gxs_df, info_df)
    df = compute_avg_and_normalize(df)
    df = pre_features(df)
    df = month_features(df)

    return df


def add_lags(df: pd.DataFrame, lags=3, rolling=5) -> pd.DataFrame:
    df = df.copy()
    for i in range(lags):
        df[f"lag{i+1}"] = df.groupby(["country", "brand_name"])["target_norm"].shift(i + 1)

    df['roll{r}_mean'.format(r=rolling)] = df.groupby(["country", "brand_name"])["target_norm"].rolling(5).mean().reset_index()['target_norm']
    df['roll{r}_std'.format(r=rolling)] = df.groupby(["country", "brand_name"])["target_norm"].rolling(5).mean().reset_index()['target_norm']

    return df


def extend_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    month_to_int = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    int_to_month = {v: k for k, v in month_to_int.items()}

    extended_rows = []
    submission_rows = []

    for (country, brand), g in df.groupby(["country", "brand_name"]):

        g = g.sort_values("months_postgx")

        # last pre-gx row (months_postgx = -1)
        last_row = g.iloc[-1]
        last_moth_postgx = last_row["months_postgx"]
        last_month_str = last_row["month"]
        last_month_int = month_to_int[last_month_str]

        # create new rows for horizon months_postgx = 0..23
        for i, h in enumerate(list(range(last_moth_postgx + 1, 24))):

            # wrap month: 1..12
            new_month_int = ((last_month_int + i) % 12) + 1
            new_month_str = int_to_month[new_month_int]

            extended_rows.append({
                "country": country,
                "brand_name": brand,
                "month": new_month_str,
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

