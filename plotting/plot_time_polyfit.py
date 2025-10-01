import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fontsize = 20
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": 18.5,
        "axes.titlesize": fontsize,
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

# Define constants for each dataset
dataset_configs = {
    "store_sales": {
        "sypnosis_size": 7773.42,
        "qtime": 1.35e-5,
        "y_lims": (-0.001, 0.045),
        "x_lims": (5400, 8100),
        "position": (0, 0),
        "title": "Store Sales Dataset",
    },
    "flights": {
        "sypnosis_size": 1941.97,
        "qtime": 1.34e-5,
        "y_lims": (-0.001, 0.03),
        "x_lims": (1000, 2200),
        "position": (0, 1),
        "title": "Flights Dataset",
    },
    "ccpp": {
        "sypnosis_size": 2914.06,
        "qtime": 1.33e-5,
        "y_lims": (-0.001, 0.03),
        "x_lims": (800, 3300),
        "position": (1, 0),
        "title": "CCPP Dataset",
    },
    "pm25": {
        "sypnosis_size": 2148.44,
        "qtime": 2.47e-5,
        "y_lims": (-0.001, 0.025),
        "x_lims": (1000, 2400),
        "position": (1, 1),
        "title": "PM25 Dataset",
    },
}

# Define colors for each method
method_colors = {"DeepMapping-R": "blue", "VerdictDB": "orange", "DBEst++": "green"}

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 11))


# Function to compute polynomial fit
def poly_fit(x, y, degree=1):
    coeffs = np.polyfit(x, y, degree)
    return coeffs


# Create a common x-axis range for ALL polynomial fits
all_sizes = []
for data_name, config in dataset_configs.items():
    df = pd.read_csv(f"results/{data_name}.csv")
    all_sizes.extend(df["size(KB)"].values)

common_x_fit = np.linspace(min(all_sizes), max(all_sizes), 100)

# Process each dataset and create a subplot
for data_name, config in dataset_configs.items():
    # Get the position for this dataset's subplot
    row, col = config["position"]
    ax = axes[row, col]

    # Load the dataset
    df = pd.read_csv(f"results/{data_name}.csv")

    # Replace DeepMapping++ with DeepMapping-R in method column
    df["method"] = df["method"].replace("DeepMapping++", "DeepMapping-R")

    # Plot each method
    for method in df["method"].unique():
        method_data = df[df["method"] == method]

        # Extract data
        x_data = method_data["size(KB)"].values
        y_data = method_data["avg_query_time"].values
        y_data_t2 = method_data["t2medium_avg_query_time"].values

        # Plot regular query time (scatter)
        ax.scatter(
            x_data,
            y_data,
            marker="o",
            color=method_colors[method],
            label=(f"{method}" if row == 0 and col == 0 else None),
            s=60,
        )

        # Fit polynomial and plot using the common x-axis
        poly_coeffs = poly_fit(x_data, y_data)
        y_fit = np.polyval(poly_coeffs, common_x_fit)
        ax.plot(
            common_x_fit,
            y_fit,
            color=method_colors[method],
            linewidth=2,
        )

        # Plot t2medium query time (scatter)
        ax.scatter(
            x_data,
            y_data_t2,
            marker="s",
            color=method_colors[method],
            label=(f"{method} (t2med)" if row == 0 and col == 0 else None),
            s=60,
        )

        # Fit polynomial for t2medium and plot using the common x-axis
        poly_coeffs_t2 = poly_fit(x_data, y_data_t2)
        y_fit_t2 = np.polyval(poly_coeffs_t2, common_x_fit)
        ax.plot(
            common_x_fit,
            y_fit_t2,
            color=method_colors[method],
            linestyle="--",
            linewidth=2,
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
        markersize=20,
        label="Synopsis Time" if row == 0 and col == 0 else None,
    )

    # Configure the subplot
    ax.set_ylim(config["y_lims"])
    ax.set_xlim(config["x_lims"])
    ax.set_xlabel("Model/Data Size (KB)")
    ax.set_ylabel("Average Query Time (s)")
    ax.set_title(config["title"], fontweight="bold")

    # Add minor ticks for better readability
    ax.minorticks_on()

    # Format y-axis to use scientific notation for small numbers
    ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Create a single legend for the entire figure
from matplotlib.lines import Line2D

# Create custom legend handles
legend_elements = []
for method, color in method_colors.items():
    # Regular method with solid line and circle marker
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=f"{method} (large-size machine)",
            markerfacecolor=color,
            markersize=8,
            linestyle="-",
            linewidth=2,
        )
    )
    # t2medium variant with dashed line and square marker
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color=color,
            label=f"{method} (small-size machine)",
            markerfacecolor=color,
            markersize=8,
            linestyle="--",
            linewidth=2,
        )
    )

# Add Synopsis line and point
legend_elements.append(
    Line2D([0], [0], color="red", linestyle="--", label="Synopsis Size")
)
legend_elements.append(
    Line2D(
        [0],
        [0],
        marker="*",
        color="red",
        label="Synopsis Time",
        markerfacecolor="red",
        markersize=15,
        linestyle="none",
    )
)

# Place the legend
fig.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.04),  # Move below the plots
    ncol=2,
    frameon=True,
)

# Adjust layout and spacing
plt.tight_layout()
plt.subplots_adjust(bottom=0.26)  # Increase bottom margin for the legend

# Save the figure
plt.savefig("plots/all_datasets_time_comparison.pdf", dpi=200, bbox_inches="tight")
plt.close()
