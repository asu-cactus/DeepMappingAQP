import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

fontsize = 20
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": 19,
        "axes.titlesize": fontsize,
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

# Set the base directory
base_dir = "results"

# Load the datasets
dm_data = pd.read_csv(os.path.join(base_dir, "pm25_DM.csv"))
nhr_data = pd.read_csv(os.path.join(base_dir, "pm25_NHR.csv"))

# Rename columns for consistency
dm_data = dm_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)
nhr_data = nhr_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)

# Cap avg_rel_err values to 1.0
dm_data["avg_rel_err"] = np.minimum(dm_data["avg_rel_err"], 1.0)
nhr_data["avg_rel_err"] = np.minimum(nhr_data["avg_rel_err"], 1.0)

# Define query percentages and line styles
query_percentages = [0.05, 0.1, 0.15]
line_styles = ["-", "--", "-."]
colors = {"DM": "blue", "NHR": "red"}
# Replace single marker per method with dictionary mapping each percentage to a marker
markers = {
    "DM": {"0.05": "o", "0.1": "v", "0.15": "^"},
    "NHR": {"0.05": "h", "0.1": "x", "0.15": "d"},
}
# Define display names mapping
display_names = {"DM": "DeepMapping-R", "NHR": "NHR"}

# Create figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# First plot DM data for all query percentages
for i, qp in enumerate(query_percentages):
    dm_filtered = dm_data[dm_data["query_percent"] == qp]
    dm_grouped = dm_filtered.groupby("size(KB)").mean().reset_index()
    dm_grouped = dm_grouped.sort_values("size(KB)")

    ax1.plot(
        dm_grouped["size(KB)"],
        dm_grouped["avg_rel_err"],
        color=colors["DM"],
        linestyle=line_styles[i],
        marker=markers["DM"][
            str(qp)
        ],  # Updated to use specific marker for this percentage
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

    ax2.plot(
        dm_grouped["size(KB)"],
        dm_grouped["avgtime"],
        color=colors["DM"],
        linestyle=line_styles[i],
        marker=markers["DM"][
            str(qp)
        ],  # Updated to use specific marker for this percentage
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

# Then plot NHR data for all query percentages
for i, qp in enumerate(query_percentages):
    nhr_filtered = nhr_data[nhr_data["query_percent"] == qp]
    nhr_filtered = nhr_filtered.sort_values("size(KB)")

    ax1.plot(
        nhr_filtered["size(KB)"],
        nhr_filtered["avg_rel_err"],
        color=colors["NHR"],
        linestyle=line_styles[i],
        marker=markers["NHR"][
            str(qp)
        ],  # Updated to use specific marker for this percentage
        label=f"{display_names['NHR']} ({int(qp*100)}%)",
    )

    ax2.plot(
        nhr_filtered["size(KB)"],
        nhr_filtered["avgtime"],
        color=colors["NHR"],
        linestyle=line_styles[i],
        marker=markers["NHR"][
            str(qp)
        ],  # Updated to use specific marker for this percentage
        label=f"{display_names['NHR']} ({int(qp*100)}%)",
    )

# Configure axes
ax1.set_xlabel("Size (KB)")
ax1.set_ylabel("Average Relative Error")
# ax1.set_title("Average Relative Error vs Size")

ax2.set_xlabel("Size (KB)")
ax2.set_ylabel("Average Query Time (s)")
# ax2.set_title("Average Query Time vs Size")

# Create custom legend with method and query percentage combinations
from matplotlib.lines import Line2D

# Create a better structured legend
handles, labels = [], []

for method, color in colors.items():
    for i, qp in enumerate(query_percentages):
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=markers[method][str(qp)],  # Updated to use specific marker
                linestyle=line_styles[i],
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")
rearrange_indices = [0, 3, 1, 4, 2, 5]
handles = [handles[i] for i in rearrange_indices]
labels = [labels[i] for i in rearrange_indices]
# Move the legend below the subplots
fig.legend(
    handles=handles,
    labels=labels,
    ncol=3,  # Increase number of columns to fit better at the bottom
    loc="lower center",  # Place at the lower center
    bbox_to_anchor=(0.5, 0.0),  # Position below the subplots
    frameon=True,
    borderaxespad=0.1,
)

# Clear the axes for the new plotting approach
ax1.clear()
ax2.clear()

# Generate common x range for smooth polynomial curves
min_size = min(dm_data["size(KB)"].min(), nhr_data["size(KB)"].min())
max_size = max(dm_data["size(KB)"].max(), nhr_data["size(KB)"].max())
x_smooth = np.linspace(min_size, max_size, 100)

# First plot DM data for all query percentages
for i, qp in enumerate(query_percentages):
    dm_filtered = dm_data[dm_data["query_percent"] == qp]
    dm_grouped = dm_filtered.groupby("size(KB)").mean().reset_index()
    dm_grouped = dm_grouped.sort_values("size(KB)")

    # Plot scatter points
    ax1.scatter(
        dm_grouped["size(KB)"],
        dm_grouped["avg_rel_err"],
        color=colors["DM"],
        marker=markers["DM"][str(qp)],
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

    ax2.scatter(
        dm_grouped["size(KB)"],
        dm_grouped["avgtime"],
        color=colors["DM"],
        marker=markers["DM"][str(qp)],
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

    # Fit polynomial and plot smooth curve
    if len(dm_grouped) >= 3:  # Need at least 3 points for 2-degree polynomial
        poly_err = np.poly1d(
            np.polyfit(dm_grouped["size(KB)"], dm_grouped["avg_rel_err"], 1)
        )
        poly_time = np.poly1d(
            np.polyfit(dm_grouped["size(KB)"], dm_grouped["avgtime"], 1)
        )

        ax1.plot(
            x_smooth,
            poly_err(x_smooth),
            color=colors["DM"],
            linestyle=line_styles[i],
        )

        ax2.plot(
            x_smooth,
            poly_time(x_smooth),
            color=colors["DM"],
            linestyle=line_styles[i],
        )

# Then plot NHR data for all query percentages
for i, qp in enumerate(query_percentages):
    nhr_filtered = nhr_data[nhr_data["query_percent"] == qp]
    nhr_filtered = nhr_filtered.sort_values("size(KB)")

    # Plot scatter points
    ax1.scatter(
        nhr_filtered["size(KB)"],
        nhr_filtered["avg_rel_err"],
        color=colors["NHR"],
        marker=markers["NHR"][str(qp)],
        label=f"{display_names['NHR']} ({int(qp*100)}%)",
    )

    ax2.scatter(
        nhr_filtered["size(KB)"],
        nhr_filtered["avgtime"],
        color=colors["NHR"],
        marker=markers["NHR"][str(qp)],
        label=f"{display_names['NHR']} ({int(qp*100)}%)",
    )

    # Fit polynomial and plot smooth curve
    if len(nhr_filtered) >= 3:  # Need at least 3 points for 2-degree polynomial
        poly_err = np.poly1d(
            np.polyfit(nhr_filtered["size(KB)"], nhr_filtered["avg_rel_err"], 1)
        )
        poly_time = np.poly1d(
            np.polyfit(nhr_filtered["size(KB)"], nhr_filtered["avgtime"], 1)
        )

        ax1.plot(
            x_smooth,
            poly_err(x_smooth),
            color=colors["NHR"],
            linestyle=line_styles[i],
        )

        ax2.plot(
            x_smooth,
            poly_time(x_smooth),
            color=colors["NHR"],
            linestyle=line_styles[i],
        )

# Configure axes
ax1.set_xlabel("Size (KB)")
ax1.set_ylabel("Average Relative Error")
ax2.set_xlabel("Size (KB)")
ax2.set_ylabel("Average Query Time (s)")

# Recreate the custom legend with the same structure as before
from matplotlib.lines import Line2D

# Create a better structured legend
handles, labels = [], []

for method, color in colors.items():
    for i, qp in enumerate(query_percentages):
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=markers[method][str(qp)],
                linestyle=line_styles[i],
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")

# Use the same rearrangement as before
rearrange_indices = [0, 3, 1, 4, 2, 5]
handles = [handles[i] for i in rearrange_indices]
labels = [labels[i] for i in rearrange_indices]

# Adjust layout with more space at the bottom for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.26)  # Increased bottom margin to accommodate the legend

# Save the figure
plt.savefig(
    "plots/pm25_dm_vs_nhr.pdf",  # Updated filename to reflect polynomial fits
    dpi=200,
    bbox_inches="tight",
)
