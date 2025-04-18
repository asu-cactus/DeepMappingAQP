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
markers = {"DM": "o", "NHP": "s"}  # Changed from NHR to NHP
# Define display names mapping
display_names = {
    "DM": "DeepMapping-R",
    "NHP": "NHP",
}  # Changed from NHR to NHP and updated the display name

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
        marker=markers["DM"],
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

    ax2.plot(
        dm_grouped["size(KB)"],
        dm_grouped["avgtime"],
        color=colors["DM"],
        linestyle=line_styles[i],
        marker=markers["DM"],
        label=f"{display_names['DM']} ({int(qp*100)}%)",
    )

# Then plot NHP data for all query percentages
for i, qp in enumerate(query_percentages):
    nhp_filtered = nhp_data[
        nhp_data["query_percent"] == qp
    ]  # Changed from nhr_filtered/nhr_data to nhp_filtered/nhp_data
    nhp_filtered = nhp_filtered.sort_values("size(KB)")

    ax1.plot(
        nhp_filtered["size(KB)"],
        nhp_filtered["avg_rel_err"],
        color=colors["NHP"],  # Changed from NHR to NHP
        linestyle=line_styles[i],
        marker=markers["NHP"],  # Changed from NHR to NHP
        label=f"{display_names['NHP']} ({int(qp*100)}%)",  # Changed from NHR to NHP
    )

    ax2.plot(
        nhp_filtered["size(KB)"],
        nhp_filtered["avgtime"],
        color=colors["NHP"],  # Changed from NHR to NHP
        linestyle=line_styles[i],
        marker=markers["NHP"],  # Changed from NHR to NHP
        label=f"{display_names['NHP']} ({int(qp*100)}%)",  # Changed from NHR to NHP
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
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker=markers[method],
                linestyle=line_styles[i],
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")
rearrange_indices = [0, 3, 1, 4, 2, 5]
handles = [handles[i] for i in rearrange_indices]
labels = [labels[i] for i in rearrange_indices]
# Move the legend to the bottom of the subplots
fig.legend(
    handles=handles,
    labels=labels,
    ncol=3,  # Changed from 3 to 2 since we only have 2 methods now
    loc="upper center",
    bbox_to_anchor=(0.5, 0.05),  # Position below the subplots
    frameon=True,
    borderaxespad=0.1,
)

# Adjust layout with more space at the bottom for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)  # Increased bottom margin to accommodate the legend

# Save the figure
plt.savefig(
    "plots/pm25_dm_vs_nhp.pdf",  # Updated filename from dm_vs_nhr to dm_vs_nhp
    dpi=200,
    bbox_inches="tight",
)
