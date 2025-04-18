import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import pandas as pd
import os

fontsize = 20
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": 20,
        "axes.titlesize": fontsize,
    }
)

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Dataset parameters
datasets = {
    "store_sales": {
        "synopsis_size": 7773.42,
        "error": 0.0008,
        "ax1_y_lims": (0.895, 0.904),
        "ax2_y_lims": (-0.02, 0.15),  # Changed from 0.035 to 0.15 to show VerdictDB
        "d": 0.015,
        "x_lims": (5400, 8100),
    },
    "flights": {
        "synopsis_size": 1941.97,
        "error": 0.0091,
        "ax1_y_lims": (0.95, 1.05),
        "ax2_y_lims": (-0.02, 0.9),  # Already good
        "d": 0.01,
        "x_lims": (1000, 2200),
    },
    "ccpp": {
        "synopsis_size": 2914.06,
        "error": 0.0014,
        "ax1_y_lims": (0.89, 0.91),
        "ax2_y_lims": (-0.02, 0.3),  # Already good
        "d": 0.008,
        "x_lims": (800, 3300),
    },
    "pm25": {
        "synopsis_size": 2148.44,
        "error": 0.0008,
        "ax1_y_lims": (0.89, 0.91),
        "ax2_y_lims": (-0.02, 0.3),  # Already good
        "d": 0.012,
        "x_lims": (1000, 2400),
    },
}

# Method colors
colors = {"DBEst++": "green", "VerdictDB": "orange", "DeepMapping++": "blue"}
titles = {
    "pm25": "PM2.5",
    "ccpp": "CCPP",
    "flights": "Flights",
    "store_sales": "Store Sales",
}


def create_broken_axis_plot(fig, gs_pos, data_name, params, show_legend=False):
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = plt.Subplot(fig, gs_pos[0])
    ax2 = plt.Subplot(fig, gs_pos[1])
    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    df = pd.read_csv(f"results/{data_name}.csv")

    for method in df["method"].unique():
        method_data = df[df["method"] == method]
        color = colors[method]
        ax1.plot(
            method_data["size(KB)"],
            method_data["avg_rel_error"],
            marker="o",
            label=method,
            linewidth=2,
            markersize=6,
            color=color,
        )
        ax2.plot(
            method_data["size(KB)"],
            method_data["avg_rel_error"],
            marker="o",
            linewidth=2,
            markersize=6,
            color=color,
        )

    # Add synopsis lines and point - remove the label from here so it doesn't create duplicate entries
    ax1.axvline(x=params["synopsis_size"], color="red", linestyle="--")
    ax2.axvline(x=params["synopsis_size"], color="red", linestyle="--")
    ax2.plot(params["synopsis_size"], params["error"], "r*", markersize=20)

    # Set limits and style
    ax1.set_ylim(params["ax1_y_lims"])
    ax2.set_ylim(params["ax2_y_lims"])
    ax1.set_xlim(params["x_lims"])
    ax2.set_xlim(params["x_lims"])

    # Configure broken axis
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Add diagonal lines
    d = params["d"]
    kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # Labels and grid
    ax2.set_xlabel("Model/Data Size (KB)")
    # Adjust the y-axis label position with labelpad parameter
    # Higher negative value moves the label higher/closer to the axis
    ax2.set_ylabel("             Average Relative Error", labelpad=20)
    ax1.set_title(f"{titles[data_name]} Dataset")

    # Legend is now handled at the figure level, not in individual subplots


# Create main figure
fig = plt.figure(figsize=(15, 9))

# Create 2x2 grid (2 rows, 2 columns)
gs_main = gridspec.GridSpec(2, 2, figure=fig)
positions = {
    "store_sales": (
        gs_main[0, 0].subgridspec(2, 1)[0],
        gs_main[0, 0].subgridspec(2, 1)[1],
    ),
    "flights": (gs_main[0, 1].subgridspec(2, 1)[0], gs_main[0, 1].subgridspec(2, 1)[1]),
    "ccpp": (gs_main[1, 0].subgridspec(2, 1)[0], gs_main[1, 0].subgridspec(2, 1)[1]),
    "pm25": (gs_main[1, 1].subgridspec(2, 1)[0], gs_main[1, 1].subgridspec(2, 1)[1]),
}

# Create all subplots
for i, (data_name, pos) in enumerate(positions.items()):
    create_broken_axis_plot(fig, pos, data_name, datasets[data_name])


custom_lines = [
    Line2D([0], [0], color=colors["DeepMapping++"], marker="o", linestyle="-"),
    Line2D([0], [0], color=colors["VerdictDB"], marker="o", linestyle="-"),
    Line2D([0], [0], color=colors["DBEst++"], marker="o", linestyle="-"),
    Line2D([0], [0], color="red", linestyle="--"),
    Line2D([0], [0], marker="*", color="red", linestyle="none", markersize=10),
]
custom_labels = [
    "DeepMapping-R",
    "VerdictDB",
    "DBEst++",
    "Synopsis Size",
    "Synopsis Error",
]

# Add the legend to the bottom of the figure
fig.legend(
    custom_lines,
    custom_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=5,
    frameon=True,
    framealpha=0.9,
)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("plots/all_datasets_error_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
