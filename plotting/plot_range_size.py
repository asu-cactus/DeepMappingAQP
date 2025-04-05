import pandas as pd
import matplotlib.pyplot as plt
import os

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
dbest_data = pd.read_csv(os.path.join(base_dir, "pm25_DBEst.csv"))
vdb_data = pd.read_csv(os.path.join(base_dir, "pm25_verdictdb_full.csv"))

# Rename columns for consistency
vdb_data = vdb_data.rename(columns={"avg_rel_error": "avg_rel_err"})
dm_data = dm_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)

# Define query percentages and line styles
query_percentages = [0.05, 0.1, 0.15]
line_styles = ["-", "--", "-."]
colors = {"DM": "blue", "DBEst": "red", "VerdictDB": "green"}
markers = {"DM": "o", "DBEst": "s", "VerdictDB": "^"}
# Define display names mapping
display_names = {"DM": "DeepMapping++", "DBEst": "DBEst++", "VerdictDB": "VerdictDB"}

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

# Then plot DBEst data for all query percentages
for i, qp in enumerate(query_percentages):
    dbest_filtered = dbest_data[dbest_data["query_percent"] == qp]
    dbest_filtered = dbest_filtered.sort_values("size(KB)")

    ax1.plot(
        dbest_filtered["size(KB)"],
        dbest_filtered["avg_rel_err"],
        color=colors["DBEst"],
        linestyle=line_styles[i],
        marker=markers["DBEst"],
        label=f"{display_names['DBEst']} ({int(qp*100)}%)",
    )

    ax2.plot(
        dbest_filtered["size(KB)"],
        dbest_filtered["avgtime"],
        color=colors["DBEst"],
        linestyle=line_styles[i],
        marker=markers["DBEst"],
        label=f"{display_names['DBEst']} ({int(qp*100)}%)",
    )

# Finally plot VerdictDB data for all query percentages
for i, qp in enumerate(query_percentages):
    vdb_filtered = vdb_data[vdb_data["query_percent"] == qp]
    vdb_filtered = vdb_filtered.sort_values("size(KB)")

    ax1.plot(
        vdb_filtered["size(KB)"],
        vdb_filtered["avg_rel_err"],
        color=colors["VerdictDB"],
        linestyle=line_styles[i],
        marker=markers["VerdictDB"],
        label=f"{display_names['VerdictDB']} ({int(qp*100)}%)",
    )

    ax2.plot(
        vdb_filtered["size(KB)"],
        vdb_filtered["avg_query_time"],
        color=colors["VerdictDB"],
        linestyle=line_styles[i],
        marker=markers["VerdictDB"],
        label=f"{display_names['VerdictDB']} ({int(qp*100)}%)",
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
                marker=markers[method],
                linestyle=line_styles[i],
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")

# Move the legend above the subplots
fig.legend(
    handles=handles,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),  # Position above the subplots
    frameon=True,
    borderaxespad=0.1,
)

# Adjust layout with more space at the top for the legend
plt.tight_layout()
plt.subplots_adjust(top=0.80)  # Increased top margin to accommodate the legend

# Save the figure
plt.savefig(
    "plots/pm25_different_ranges.pdf",
    dpi=300,
    bbox_inches="tight",
)
