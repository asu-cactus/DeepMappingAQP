import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define constants for each dataset
dataset_configs = {
    "store_sales": {
        "sypnosis_size": 7773.42,
        "qtime": 1.35e-5,
        "y_lims": (0, 0.045),
        "x_lims": (5400, 8100),
        "position": (0, 0),
        "title": "Store Sales",
    },
    "flights": {
        "sypnosis_size": 1941.97,
        "qtime": 1.34e-5,
        "y_lims": (0, 0.03),
        "x_lims": (1000, 2200),
        "position": (0, 1),
        "title": "Flights",
    },
    "ccpp": {
        "sypnosis_size": 2914.06,
        "qtime": 1.33e-5,
        "y_lims": (0, 0.03),
        "x_lims": (800, 3300),
        "position": (1, 0),
        "title": "CCPP",
    },
    "pm25": {
        "sypnosis_size": 2148.44,
        "qtime": 2.47e-5,
        "y_lims": (0, 0.025),
        "x_lims": (1000, 2400),
        "position": (1, 1),
        "title": "PM2.5",
    },
}

# Define colors for each method
method_colors = {"DeepMapping++": "blue", "VerdictDB": "orange", "DBEst++": "green"}

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Process each dataset and create a subplot
for data_name, config in dataset_configs.items():
    # Get the position for this dataset's subplot
    row, col = config["position"]
    ax = axes[row, col]

    # Load the dataset
    df = pd.read_csv(f"results/{data_name}.csv")

    # Plot each method
    for method in df["method"].unique():
        method_data = df[df["method"] == method]

        # Plot regular query time
        ax.plot(
            method_data["size(KB)"],
            method_data["avg_query_time"],
            marker="o",
            color=method_colors[method],
            label=(
                f"{method}" if row == 0 and col == 0 else None
            ),  # Only add label in first subplot
            linewidth=2,
            markersize=6,
        )

        # Plot t2medium query time
        ax.plot(
            method_data["size(KB)"],
            method_data["t2medium_avg_query_time"],
            marker="s",
            linestyle="--",
            color=method_colors[method],
            label=(
                f"{method} (t2medium)" if row == 0 and col == 0 else None
            ),  # Only add label in first subplot
            linewidth=2,
            markersize=6,
        )

    # Add vertical line and point for synopsis
    ax.axvline(
        x=config["sypnosis_size"],
        color="red",
        linestyle="--",
        label="Synopsis" if row == 0 and col == 0 else None,
    )
    ax.plot(
        config["sypnosis_size"],
        config["qtime"],
        "r*",
        markersize=10,
        label="Synopsis Time" if row == 0 and col == 0 else None,
    )

    # Configure the subplot
    ax.set_ylim(config["y_lims"])
    ax.set_xlim(config["x_lims"])
    ax.set_xlabel("Model/Data Size (KB)", fontsize=12)
    ax.set_ylabel("Average Query Time (s)", fontsize=12)
    ax.set_title(config["title"], fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add minor ticks for better readability
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)

    # Format y-axis to use scientific notation for small numbers
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Create a single legend for the entire figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),  # Move higher above the plots
    ncol=3,
    fontsize=12,
    frameon=True,
)

# Adjust layout and spacing
plt.tight_layout()
plt.subplots_adjust(top=0.93)  # Increase top margin for the legend

# Save the figure
plt.savefig("plots/all_datasets_time_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
