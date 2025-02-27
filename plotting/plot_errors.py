import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd

import argparse
import pdb

parser = argparse.ArgumentParser(description="Plot error comparison")
parser.add_argument("--data_name", type=str, required=True, help="Data name")
args = parser.parse_args()

# Read the combined CSV file
data_name = args.data_name

if data_name == "store_sales":
    sypnosis_size = 7773.42
    error = 0.0008
    ax1_y_lims = (0.895, 0.902)  # Tighter range for DBEst++
    ax2_y_lims = (0, 0.035)  # Slightly higher to show all points
    d = 0.015  # Size of diagonal lines
    x_lims = (1500, 8500)  # Focus on relevant x-range
elif data_name == "flights":
    sypnosis_size = 1941.97
    error = 0.0091
    ax1_y_lims = (0.99, 1.01)  # Tighter range around 1.0
    ax2_y_lims = (0, 0.35)  # Adjusted to show all errors clearly
    d = 0.01  # Smaller diagonals
    x_lims = (1000, 2500)  # Focus on relevant x-range
elif data_name == "ccpp":
    sypnosis_size = 2914.06
    error = 0.0014
    ax1_y_lims = (0.889, 1.005)  # Very tight range for DBEst++
    ax2_y_lims = (0, 0.015)  # Show small errors clearly
    d = 0.008  # Very small diagonals
    x_lims = (700, 3000)  # Focus on relevant x-range
elif data_name == "pm25":
    sypnosis_size = 2148.44
    error = 0.0008
    ax1_y_lims = (0.897, 0.900)  # Tight range for DBEst++
    ax2_y_lims = (0, 0.065)  # Show all points with margin
    d = 0.012  # Medium diagonals
    x_lims = (1000, 2600)  # Focus on relevant x-range
else:
    raise ValueError(f"No support for {data_name} for 1D input")
df = pd.read_csv(f"results/{data_name}.csv")

# Create figure with broken y-axis
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# Plot on both axes
for method in df["method"].unique():
    method_data = df[df["method"] == method]
    ax1.plot(
        method_data["size(KB)"],
        method_data["avg_rel_error"],
        marker="o",
        label=method,
        linewidth=2,
        markersize=6,
    )
    ax2.plot(
        method_data["size(KB)"],
        method_data["avg_rel_error"],
        marker="o",
        label=method,
        linewidth=2,
        markersize=6,
    )

# Add vertical lines for synopsis size
ax1.axvline(x=sypnosis_size, color="red", linestyle="--", label="Synopsis")
ax2.axvline(x=sypnosis_size, color="red", linestyle="--", label="Synopsis")

# Add special point for synopsis performance
ax2.plot(sypnosis_size, error, "r*", markersize=10, label="Synopsis Error")

# Set different scales for the two plots
ax1.set_ylim(ax1_y_lims[0], ax1_y_lims[1])  # Upper subplot for DBest++
ax2.set_ylim(
    ax2_y_lims[0], ax2_y_lims[1]
)  # Lower subplot for DeepMapping++ and VerdictDB

# Hide the spines between ax1 and ax2
ax1.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Add broken axis indicators

kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax1.set_xlabel("Model/Data Size (KB)", fontsize=12)
ax1.set_ylabel("Average Relative Error", fontsize=12)
ax1.set_title("Model/Data Size vs. Average Relative Error Comparison", fontsize=14)
ax1.grid(True, linestyle="--", alpha=0.7)
ax1.legend()

ax1.set_xlim(x_lims)
ax2.set_xlim(x_lims)
ax2.grid(True, linestyle="--", alpha=0.7)  # Add grid to bottom plot too

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(f"plots/{data_name}_error_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
