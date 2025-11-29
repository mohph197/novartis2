import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def merge_pred_true(df_pred: pd.DataFrame, df_true: pd.DataFrame) -> pd.DataFrame:
    return df_pred.merge(
        df_true,
        how='left',
        on=['country', 'brand_name', 'months_postgx'],
        validate='one_to_one'
    )


def plot_static_samples(df_pred: pd.DataFrame, df_true: pd.DataFrame, n_samples=9, title_prefix=""):
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

        ax.set_title(f'{title_prefix}{combo}')
        ax.set_xlabel('Months Post GX')
        ax.set_ylabel('Units')
        ax.legend(loc='best', fontsize='small')
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots (if n_samples isn't perfectly divisible by 3)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()


def plot_interactive_comparison(df_pred: pd.DataFrame, df_true: pd.DataFrame, title_prefix=""):
    """
    Creates a single interactive Plotly chart with a dropdown menu
    to switch between all country-brand combinations.
    """
    df = merge_pred_true(df_pred, df_true)

    print("Generating Interactive Plotly Chart...")

    # Create combo_id if it doesn't exist
    if 'combo_id' not in df.columns:
        df = df.copy()
        df['combo_id'] = df['country'].astype(str) + " - " + df['brand_name'].astype(str)

    unique_combos = df['combo_id'].unique()

    if len(unique_combos) == 0:
        print("No data to plot.")
        return

    fig = go.Figure()

    # Add traces for ALL combinations (Hidden by default)
    # We add them in pairs: (Volume, Value True)
    for combo in unique_combos:
        subset = df[df['combo_id'] == combo]

        # Trace 1: Prediction (Volume)
        fig.add_trace(go.Scatter(
            x=subset['months_postgx'],
            y=subset['volume'],
            name='Prediction',
            visible=False,
            line=dict(color='#1f77b4', width=2)
        ))

        # Trace 2: Actual (Value True)
        fig.add_trace(go.Scatter(
            x=subset['months_postgx'],
            y=subset['vol_true'],
            name='Actual',
            visible=False,
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))

    # Make the first combination visible initially
    if len(fig.data) > 0:
        fig.data[0].visible = True
        fig.data[1].visible = True

    # Create Dropdown Menu
    steps = []
    for i, combo in enumerate(unique_combos):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"{title_prefix}{combo}"}],
            label=combo
        )
        # Toggle the specific pair of traces for this combo
        # Since we added them in pairs, indices are 2*i and 2*i + 1
        step["args"][0]["visible"][2*i] = True     # Volume
        step["args"][0]["visible"][2*i + 1] = True # Value True
        steps.append(step)

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=steps)],
        title=f"{title_prefix}{unique_combos[0]}",
        xaxis_title="Months Post GX",
        yaxis_title="Units",
        template="plotly_white"
    )

    fig.show()