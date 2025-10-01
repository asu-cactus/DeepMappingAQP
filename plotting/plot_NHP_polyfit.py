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
nhp_data = pd.read_csv(
    os.path.join(base_dir, "pm25_NHP.csv")
)  # Changed from NHR to NHP

# Rename columns for consistency
dm_data = dm_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)
nhp_data = nhp_data.rename(  # Changed from nhr_data to nhp_data
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)

# Cap avg_rel_err values to 1.0 (any value > 1.0 will be set to 1.0)
print(f"DM data before capping: {dm_data['avg_rel_err'].max():.4f} max error")
print(
    f"NHP data before capping: {nhp_data['avg_rel_err'].max():.4f} max error"
)  # Changed from NHR to NHP

dm_data["avg_rel_err"] = np.minimum(dm_data["avg_rel_err"], 1.0)
nhp_data["avg_rel_err"] = np.minimum(
    nhp_data["avg_rel_err"], 1.0
)  # Changed from nhr_data to nhp_data

print(f"DM data after capping: {dm_data['avg_rel_err'].max():.4f} max error")
print(
    f"NHP data after capping: {nhp_data['avg_rel_err'].max():.4f} max error"
)  # Changed from NHR to NHP

# Define query percentages and line styles
query_percentages = [0.05, 0.1, 0.15]
line_styles = ["-", "--", "-."]
colors = {"DM": "blue", "NHP": "red"}  # Changed from NHR to NHP
# Updated markers to be specific for each query percentage
dm_markers = {"0.05": "o", "0.1": "v", "0.15": "^"}
nhp_markers = {"0.05": "h", "0.1": "x", "0.15": "d"}
# Define display names mapping
display_names = {
    "DM": "DeepMapping-R",
    "NHP": "NHP",
}  # Changed from NHR to NHP and updated the display name

# Create figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Changed from (15, 6) to (15, 5)

# Create common x-range for polynomial fitting
x_min = min(dm_data["size(KB)"].min(), nhp_data["size(KB)"].min())
x_max = max(dm_data["size(KB)"].max(), nhp_data["size(KB)"].max())
x_fit = np.linspace(x_min, x_max, 100)

# First plot DM data for all query percentages
for i, qp in enumerate(query_percentages):
    dm_filtered = dm_data[dm_data["query_percent"] == qp]
    dm_grouped = dm_filtered.groupby("size(KB)").mean().reset_index()
    dm_grouped = dm_grouped.sort_values("size(KB)")

    # Scatter plot for actual data points
    ax1.scatter(
        dm_grouped["size(KB)"],
        dm_grouped["avg_rel_err"],
        color=colors["DM"],
        marker=dm_markers[str(qp)],  # Updated to use query-specific marker
        s=60,
        alpha=0.7,
    )

    # Polynomial fitting for error
    poly_err = np.polyfit(dm_grouped["size(KB)"], dm_grouped["avg_rel_err"], 2)
    poly_err_fn = np.poly1d(poly_err)
    ax1.plot(
        x_fit,
        poly_err_fn(x_fit),
        color=colors["DM"],
        linestyle=line_styles[i],
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

    # Scatter plot for query time
    ax2.scatter(
        dm_grouped["size(KB)"],
        dm_grouped["avgtime"],
        color=colors["DM"],
        marker=dm_markers[str(qp)],  # Updated to use query-specific marker
        s=60,
        alpha=0.7,
    )

    # Polynomial fitting for time
    poly_time = np.polyfit(dm_grouped["size(KB)"], dm_grouped["avgtime"], 2)
    poly_time_fn = np.poly1d(poly_time)
    ax2.plot(
        x_fit,
        poly_time_fn(x_fit),
        color=colors["DM"],
        linestyle=line_styles[i],
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

# Then plot NHP data for all query percentages
for i, qp in enumerate(query_percentages):
    nhp_filtered = nhp_data[nhp_data["query_percent"] == qp]
    nhp_filtered = nhp_filtered.sort_values("size(KB)")

    # Scatter plot for error
    ax1.scatter(
        nhp_filtered["size(KB)"],
        nhp_filtered["avg_rel_err"],
        color=colors["NHP"],
        marker=nhp_markers[str(qp)],  # Updated to use query-specific marker
        s=60,
        alpha=0.7,
    )

    # Polynomial fitting for error
    poly_err = np.polyfit(nhp_filtered["size(KB)"], nhp_filtered["avg_rel_err"], 2)
    poly_err_fn = np.poly1d(poly_err)
    ax1.plot(
        x_fit,
        poly_err_fn(x_fit),
        color=colors["NHP"],
        linestyle=line_styles[i],
        label=f"{display_names['NHP']} ({int(qp*100)}%)",
    )

    # Scatter plot for query time
    ax2.scatter(
        nhp_filtered["size(KB)"],
        nhp_filtered["avgtime"],
        color=colors["NHP"],
        marker=nhp_markers[str(qp)],  # Updated to use query-specific marker
        s=60,
        alpha=0.7,
    )

    # Polynomial fitting for time
    poly_time = np.polyfit(nhp_filtered["size(KB)"], nhp_filtered["avgtime"], 2)
    poly_time_fn = np.poly1d(poly_time)
    ax2.plot(
        x_fit,
        poly_time_fn(x_fit),
        color=colors["NHP"],
        linestyle=line_styles[i],
        label=f"{display_names['NHP']} ({int(qp*100)}%)",
    )

# Configure axes
ax1.set_xlabel("Size (KB)")
ax1.set_ylabel("Average Relative Error")
ax1.set_ylim(0, 0.8)

ax2.set_xlabel("Size (KB)")
ax2.set_ylabel("Average Query Time (s)")
ax2.set_yscale("log")  # Using log scale for query time to better show differences

# Create custom legend with method and query percentage combinations
from matplotlib.lines import Line2D

# Create a better structured legend
handles, labels = [], []

for method, color in colors.items():
    for i, qp in enumerate(query_percentages):
        marker = dm_markers[str(qp)] if method == "DM" else nhp_markers[str(qp)]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=marker,
                linestyle=line_styles[i],
                markersize=6,  # Added to ensure markers are visible in smaller plot
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")
rearrange_indices = [0, 3, 1, 4, 2, 5]
handles = [handles[i] for i in rearrange_indices]
labels = [labels[i] for i in rearrange_indices]

# Move the legend to the bottom of the subplots with adjusted parameters for smaller height
fig.legend(
    handles=handles,
    labels=labels,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.03),  # Adjusted from 0.05 to 0.03
    frameon=True,
    borderaxespad=0.1,
    fontsize=18,  # Added to make legend text slightly smaller to fit better
)

# Adjust layout with more space at the bottom for the legend
plt.tight_layout()
plt.subplots_adjust(
    bottom=0.16
)  # Adjusted from 0.16 to 0.2 to make more room for legend

# Save the figure
plt.savefig(
    "plots/pm25_dm_vs_nhp.pdf",
    dpi=200,
    bbox_inches="tight",
)
