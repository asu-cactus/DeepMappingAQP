import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Plotting script for different ranges")
parser.add_argument(
    "--data_name",
    type=str,
    required=True,
    help="Name of the dataset to plot",
)
args = parser.parse_args()
# Set the data name based on the argument or default value
data_name = args.data_name
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
dm_data = pd.read_csv(os.path.join(base_dir, f"{data_name}_DM.csv"))
dbest_data = pd.read_csv(os.path.join(base_dir, f"{data_name}_DBEst.csv"))
vdb_data = pd.read_csv(os.path.join(base_dir, f"{data_name}_verdictdb_full.csv"))

dm_data["avg_rel_error"] = np.minimum(dm_data["avg_rel_error"], 1.0)
dbest_data["avg_rel_err"] = np.minimum(dbest_data["avg_rel_err"], 1.0)
vdb_data["avg_rel_error"] = np.minimum(vdb_data["avg_rel_error"], 1.0)

# Rename columns for consistency
vdb_data = vdb_data.rename(columns={"avg_rel_error": "avg_rel_err"})
dm_data = dm_data.rename(
    columns={"avg_rel_error": "avg_rel_err", "avg_query_time": "avgtime"}
)

# Define query percentages and line styles
query_percentages = [0.05, 0.1, 0.15]
line_styles = ["-", "--", "-."]
colors = {"DM": "blue", "VerdictDB": "orange", "DBEst": "green"}
markers = {"DM": "o", "DBEst": "s", "VerdictDB": "^"}
# Define display names mapping
display_names = {"DM": "DeepMapping-R", "VerdictDB": "VerdictDB", "DBEst": "DBEst++"}

# Create figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Create a common set of x points for the polynomial fits
all_sizes = (
    pd.concat(
        [
            dm_data["size(KB)"].drop_duplicates(),
            dbest_data["size(KB)"].drop_duplicates(),
            vdb_data["size(KB)"].drop_duplicates(),
        ]
    )
    .drop_duplicates()
    .sort_values()
    .values
)

x_fit = np.linspace(min(all_sizes), max(all_sizes), 100)

# First plot DM data for all query percentages
for i, qp in enumerate(query_percentages):
    dm_filtered = dm_data[dm_data["query_percent"] == qp]
    dm_grouped = dm_filtered.groupby("size(KB)").mean().reset_index()
    dm_grouped = dm_grouped.sort_values("size(KB)")

    # Scatter plot for data points
    ax1.scatter(
        dm_grouped["size(KB)"],
        dm_grouped["avg_rel_err"],
        color=colors["DM"],
        marker=markers["DM"],
        label=f"{display_names['DM']} ({int(qp*100)}%) points",
    )

    # Polynomial fit (degree 2) for error
    if len(dm_grouped) >= 3:  # Need at least 3 points for quadratic fit
        poly_coeff = np.polyfit(dm_grouped["size(KB)"], dm_grouped["avg_rel_err"], 2)
        poly_func = np.poly1d(poly_coeff)
        ax1.plot(
            x_fit,
            poly_func(x_fit),
            color=colors["DM"],
            linestyle=line_styles[i],
            label=f"{display_names['DM']} ({int(qp*100)}%) fit",
        )

    # Scatter plot for time
    ax2.scatter(
        dm_grouped["size(KB)"],
        dm_grouped["avgtime"],
        color=colors["DM"],
        marker=markers["DM"],
        label=f"{display_names['DM']} ({int(qp*100)}%) points",
    )

    # Polynomial fit (degree 2) for time
    if len(dm_grouped) >= 3:
        poly_coeff_time = np.polyfit(dm_grouped["size(KB)"], dm_grouped["avgtime"], 2)
        poly_func_time = np.poly1d(poly_coeff_time)
        ax2.plot(
            x_fit,
            poly_func_time(x_fit),
            color=colors["DM"],
            linestyle=line_styles[i],
            label=f"{display_names['DM']} ({int(qp*100)}%) fit",
        )

# Then plot DBEst data for all query percentages
for i, qp in enumerate(query_percentages):
    dbest_filtered = dbest_data[dbest_data["query_percent"] == qp]
    dbest_filtered = dbest_filtered.sort_values("size(KB)")

    # Scatter plot for data points
    ax1.scatter(
        dbest_filtered["size(KB)"],
        dbest_filtered["avg_rel_err"],
        color=colors["DBEst"],
        marker=markers["DBEst"],
        label=f"{display_names['DBEst']} ({int(qp*100)}%) points",
    )

    # Polynomial fit (degree 2) for error
    if len(dbest_filtered) >= 3:
        poly_coeff = np.polyfit(
            dbest_filtered["size(KB)"], dbest_filtered["avg_rel_err"], 2
        )
        poly_func = np.poly1d(poly_coeff)
        ax1.plot(
            x_fit,
            poly_func(x_fit),
            color=colors["DBEst"],
            linestyle=line_styles[i],
            label=f"{display_names['DBEst']} ({int(qp*100)}%) fit",
        )

    # Scatter plot for time
    ax2.scatter(
        dbest_filtered["size(KB)"],
        dbest_filtered["avgtime"],
        color=colors["DBEst"],
        marker=markers["DBEst"],
        label=f"{display_names['DBEst']} ({int(qp*100)}%) points",
    )

    # Polynomial fit (degree 2) for time
    if len(dbest_filtered) >= 3:
        poly_coeff_time = np.polyfit(
            dbest_filtered["size(KB)"], dbest_filtered["avgtime"], 2
        )
        poly_func_time = np.poly1d(poly_coeff_time)
        ax2.plot(
            x_fit,
            poly_func_time(x_fit),
            color=colors["DBEst"],
            linestyle=line_styles[i],
            label=f"{display_names['DBEst']} ({int(qp*100)}%) fit",
        )

# Finally plot VerdictDB data for all query percentages
for i, qp in enumerate(query_percentages):
    vdb_filtered = vdb_data[vdb_data["query_percent"] == qp]
    vdb_filtered = vdb_filtered.sort_values("size(KB)")

    # Scatter plot for data points
    ax1.scatter(
        vdb_filtered["size(KB)"],
        vdb_filtered["avg_rel_err"],
        color=colors["VerdictDB"],
        marker=markers["VerdictDB"],
        label=f"{display_names['VerdictDB']} ({int(qp*100)}%) points",
    )

    # Polynomial fit (degree 2) for error
    if len(vdb_filtered) >= 3:
        poly_coeff = np.polyfit(
            vdb_filtered["size(KB)"], vdb_filtered["avg_rel_err"], 2
        )
        poly_func = np.poly1d(poly_coeff)
        ax1.plot(
            x_fit,
            poly_func(x_fit),
            color=colors["VerdictDB"],
            linestyle=line_styles[i],
            label=f"{display_names['VerdictDB']} ({int(qp*100)}%) fit",
        )

    # Scatter plot for time
    ax2.scatter(
        vdb_filtered["size(KB)"],
        vdb_filtered["avg_query_time"],
        color=colors["VerdictDB"],
        marker=markers["VerdictDB"],
        label=f"{display_names['VerdictDB']} ({int(qp*100)}%) points",
    )

    # Polynomial fit (degree 2) for time
    if len(vdb_filtered) >= 3:
        poly_coeff_time = np.polyfit(
            vdb_filtered["size(KB)"], vdb_filtered["avg_query_time"], 2
        )
        poly_func_time = np.poly1d(poly_coeff_time)
        ax2.plot(
            x_fit,
            poly_func_time(x_fit),
            color=colors["VerdictDB"],
            linestyle=line_styles[i],
            label=f"{display_names['VerdictDB']} ({int(qp*100)}%) fit",
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

# Follow the specified ordering: DeepMapping-R, VerdictDB, DBEst++
methods_order = ["DM", "VerdictDB", "DBEst"]
for method in methods_order:
    for i, qp in enumerate(query_percentages):
        # Add line for fit
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[method],
                linestyle=line_styles[i],
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")

        # # Add marker for data points
        # handles.append(
        #     Line2D(
        #         [0],
        #         [0],
        #         color=colors[method],
        #         marker=markers[method],
        #         linestyle="none",
        #         label=f"{display_names[method]} ({int(qp*100)}%) points",
        #     )
        # )
        # labels.append(f"{display_names[method]} ({int(qp*100)}%) points")

# Move the legend below the subplots
fig.legend(
    handles=handles,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),  # Position below the subplots
    frameon=True,
    borderaxespad=0.1,
)

# Adjust layout with more space at the bottom for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # Increased bottom margin to accommodate the legend

# Save the figure
plt.savefig(
    f"plots/{data_name}_different_ranges_polyfit.png",
    dpi=200,
    bbox_inches="tight",
)
