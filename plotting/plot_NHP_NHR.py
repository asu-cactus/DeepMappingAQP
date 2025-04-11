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
nhp_data = pd.read_csv(os.path.join(base_dir, "pm25_NHP.csv"))
nhr_data = pd.read_csv(os.path.join(base_dir, "pm25_NHR.csv"))

# Rename columns for consistency
dm_data = dm_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)
nhp_data = nhp_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)
nhr_data = nhr_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)

# Cap avg_rel_err values to 1.0 (any value > 1.0 will be set to 1.0)
print(f"DM data before capping: {dm_data['avg_rel_err'].max():.4f} max error")
print(f"NHP data before capping: {nhp_data['avg_rel_err'].max():.4f} max error")
print(f"NHR data before capping: {nhr_data['avg_rel_err'].max():.4f} max error")

dm_data["avg_rel_err"] = np.minimum(dm_data["avg_rel_err"], 1.0)
nhp_data["avg_rel_err"] = np.minimum(nhp_data["avg_rel_err"], 1.0)
nhr_data["avg_rel_err"] = np.minimum(nhr_data["avg_rel_err"], 1.0)

print(f"DM data after capping: {dm_data['avg_rel_err'].max():.4f} max error")
print(f"NHP data after capping: {nhp_data['avg_rel_err'].max():.4f} max error")
print(f"NHR data after capping: {nhr_data['avg_rel_err'].max():.4f} max error")

# Define query percentages and line styles
query_percentages = [0.05, 0.1, 0.15]
line_styles = ["-", "--", "-."]
colors = {"DM": "blue", "NHP": "red", "NHR": "green"}
markers = {"DM": "o", "NHP": "s", "NHR": "^"}
# Define display names mapping
display_names = {
    "DM": "DeepMapping++",
    "NHP": "Neural Histogram - Point",
    "NHR": "Neural Histogram - Range",
}

# Create figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

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
    nhp_filtered = nhp_data[nhp_data["query_percent"] == qp]
    nhp_filtered = nhp_filtered.sort_values("size(KB)")

    ax1.plot(
        nhp_filtered["size(KB)"],
        nhp_filtered["avg_rel_err"],
        color=colors["NHP"],
        linestyle=line_styles[i],
        marker=markers["NHP"],
        label=f"{display_names['NHP']} ({int(qp*100)}%)",
    )

    ax2.plot(
        nhp_filtered["size(KB)"],
        nhp_filtered["avgtime"],
        color=colors["NHP"],
        linestyle=line_styles[i],
        marker=markers["NHP"],
        label=f"{display_names['NHP']} ({int(qp*100)}%)",
    )

# Plot NHR data for all query percentages
for i, qp in enumerate(query_percentages):
    nhr_filtered = nhr_data[nhr_data["query_percent"] == qp]
    nhr_filtered = nhr_filtered.sort_values("size(KB)")

    ax1.plot(
        nhr_filtered["size(KB)"],
        nhr_filtered["avg_rel_err"],
        color=colors["NHR"],
        linestyle=line_styles[i],
        marker=markers["NHR"],
        label=f"{display_names['NHR']} ({int(qp*100)}%)",
    )

    ax2.plot(
        nhr_filtered["size(KB)"],
        nhr_filtered["avgtime"],
        color=colors["NHR"],
        linestyle=line_styles[i],
        marker=markers["NHR"],
        label=f"{display_names['NHR']} ({int(qp*100)}%)",
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

# Move the legend above the subplots
fig.legend(
    handles=handles,
    ncol=3,  # Changed to 3 since we now have 3 methods
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),  # Position above the subplots
    frameon=True,
    borderaxespad=0.1,
)

# Adjust layout with more space at the top for the legend
plt.tight_layout()
plt.subplots_adjust(top=0.78)  # Increased top margin to accommodate the legend

# Save the figure
plt.savefig(
    "plots/pm25_dm_vs_nhp_nhr.pdf",  # Updated filename to include all three methods
    dpi=300,
    bbox_inches="tight",
)
