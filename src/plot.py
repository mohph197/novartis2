import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np


def merge_pred_true(df_pred: pd.DataFrame, df_true: pd.DataFrame) -> pd.DataFrame:
    return df_pred.merge(
        df_pred,
        how='left',
        on=['country', 'brand_name', 'months_postgx'],
        validate='one_to_one'
    )


def plot_static_samples(df_pred: pd.DataFrame, df_true: pd.DataFrame, n_samples=9):
    """
    Selects 'n_samples' random country-brand combinations and plots them
    in a grid layout with 3 plots per row using Matplotlib.
    """
    df = merge_pred_true(df_pred, df_true)

    # Create combo_id if it doesn't exist
    if 'combo_id' not in df.columns:
        df = df.copy()
        df['combo_id'] = df['country'].astype(str) + " - " + df['brand_name'].astype(str)

    unique_combos = df['combo_id'].unique()

    # Handle case where we have fewer combos than requested samples
    n = min(n_samples, len(unique_combos))
    if n == 0:
        print("No combinations found in DataFrame.")
        return

    # Randomly select combos to plot
    selected_combos = np.random.choice(unique_combos, n, replace=False)
    print(f"Generating {n} static plots for: {selected_combos}")

    # Calculate grid dimensions (3 columns fixed)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols  # Ceiling division

    # Create figure with subplots
    # Adjust figure height based on number of rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), constrained_layout=True)

    # Flatten axes array for easy iteration (handles 1D and 2D arrays)
    # If n=1, axes is a single object, so we wrap it in a list
    if n == 1:
        axes: list[Axes] = [axes]
    else:
        axes: list[Axes] = axes.flatten()

    for i, combo in enumerate(selected_combos):
        ax = axes[i]
        subset = df[df['combo_id'] == combo]

        # Plot Prediction (Volume)
        ax.plot(subset['months_postgx'], subset['volume'],
                 color='tab:blue', label='Prediction (Volume)', linewidth=2)

        # Plot Actuals (Value True)
        ax.plot(subset['months_postgx'], subset['vol_true'],
                 color='tab:orange', linestyle='--', label='Actual (Value True)', linewidth=2)

        ax.set_title(f'{combo}')
        ax.set_xlabel('Months Post GX')
        ax.set_ylabel('Units')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (if n_samples isn't perfectly divisible by 3)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()
