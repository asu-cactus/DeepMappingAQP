import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D

# Set font size and style
fontsize = 24
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "axes.titlesize": fontsize,
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)

# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams['axes.labelweight'] = 'bold'

# Set the base directory
base_dir = "results"

# Define display names mapping
display_names = {"DM": "DeepMapping-R", "VerdictDB": "VerdictDB", "DBEst": "DBEst++"}
colors = {"DM": "blue", "VerdictDB": "orange", "DBEst": "green"}
# Replace the markers dictionary to use different markers for percentages instead of methods
query_percentages = [0.05, 0.1, 0.15]
markers = {0.05: "o", 0.1: "x", 0.15: "d"}  # Specific markers for each percentage
line_styles = ["-", "--", "-."]
methods_order = ["DM", "VerdictDB", "DBEst"]

# Create a 2x2 figure layout
fig, axes = plt.subplots(2, 2, figsize=(20, 14))


# Function to load and process data for a dataset
def load_dataset(data_name):
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

    return dm_data, dbest_data, vdb_data


# Function to plot data on axes
def plot_dataset(dm_data, dbest_data, vdb_data, ax_error, ax_time, dataset_title):
    # Create a common x-axis for polynomial fitting
    # all_sizes = np.concatenate(
    #     [
    #         dm_data["size(KB)"].unique(),
    #         dbest_data["size(KB)"].unique(),
    #         vdb_data["size(KB)"].unique(),
    #     ]
    # )
    # poly_x = np.linspace(
    #     min(all_sizes), max(all_sizes), 100
    # )  # 100 points for smooth curve

    dm_sizes = dm_data["size(KB)"].unique()
    poly_x = np.linspace(
        min(dm_sizes), max(dm_sizes), 100
    )  # 100 points for smooth curve

    # Plot DM data
    for i, qp in enumerate(query_percentages):
        dm_filtered = dm_data[dm_data["query_percent"] == qp]
        dm_grouped = dm_filtered.groupby("size(KB)").mean().reset_index()
        dm_grouped = dm_grouped.sort_values("size(KB)")

        # Scatter plot for original points - use marker based on query percentage
        ax_error.scatter(
            dm_grouped["size(KB)"],
            dm_grouped["avg_rel_err"],
            color=colors["DM"],
            marker=markers[qp],
            s=80,  # Size of markers
        )

        ax_time.scatter(
            dm_grouped["size(KB)"],
            dm_grouped["avgtime"],
            color=colors["DM"],
            marker=markers[qp],
            s=80,
        )

        # Polynomial fitting for error
        if len(dm_grouped) > 2:  # Need at least 3 points for 2-degree polynomial
            poly_error = np.poly1d(
                np.polyfit(dm_grouped["size(KB)"], dm_grouped["avg_rel_err"], 2)
            )
            ax_error.plot(
                poly_x,
                poly_error(poly_x),
                color=colors["DM"],
                linestyle=line_styles[i],
            )

            # Polynomial fitting for time
            poly_time = np.poly1d(
                np.polyfit(dm_grouped["size(KB)"], dm_grouped["avgtime"], 2)
            )
            ax_time.plot(
                poly_x,
                poly_time(poly_x),
                color=colors["DM"],
                linestyle=line_styles[i],
            )

    # Plot DBEst data
    for i, qp in enumerate(query_percentages):
        dbest_filtered = dbest_data[dbest_data["query_percent"] == qp]
        dbest_filtered = dbest_filtered.sort_values("size(KB)")

        # Scatter plot for original points - use marker based on query percentage
        ax_error.scatter(
            dbest_filtered["size(KB)"],
            dbest_filtered["avg_rel_err"],
            color=colors["DBEst"],
            marker=markers[qp],
            s=80,
        )

        ax_time.scatter(
            dbest_filtered["size(KB)"],
            dbest_filtered["avgtime"],
            color=colors["DBEst"],
            marker=markers[qp],
            s=80,
        )

        # Polynomial fitting for error
        if len(dbest_filtered) > 2:  # Need at least 3 points for 2-degree polynomial
            poly_error = np.poly1d(
                np.polyfit(dbest_filtered["size(KB)"], dbest_filtered["avg_rel_err"], 2)
            )
            ax_error.plot(
                poly_x,
                poly_error(poly_x),
                color=colors["DBEst"],
                linestyle=line_styles[i],
            )

            # Polynomial fitting for time
            poly_time = np.poly1d(
                np.polyfit(dbest_filtered["size(KB)"], dbest_filtered["avgtime"], 2)
            )
            ax_time.plot(
                poly_x,
                poly_time(poly_x),
                color=colors["DBEst"],
                linestyle=line_styles[i],
            )

    # Plot VerdictDB data
    for i, qp in enumerate(query_percentages):
        vdb_filtered = vdb_data[vdb_data["query_percent"] == qp]
        vdb_filtered = vdb_filtered.sort_values("size(KB)")

        # Scatter plot for original points - use marker based on query percentage
        ax_error.scatter(
            vdb_filtered["size(KB)"],
            vdb_filtered["avg_rel_err"],
            color=colors["VerdictDB"],
            marker=markers[qp],
            s=80,
        )

        ax_time.scatter(
            vdb_filtered["size(KB)"],
            vdb_filtered["avg_query_time"],
            color=colors["VerdictDB"],
            marker=markers[qp],
            s=80,
        )

        # Polynomial fitting for error
        if len(vdb_filtered) > 2:  # Need at least 3 points for 2-degree polynomial
            poly_error = np.poly1d(
                np.polyfit(vdb_filtered["size(KB)"], vdb_filtered["avg_rel_err"], 2)
            )
            ax_error.plot(
                poly_x,
                poly_error(poly_x),
                color=colors["VerdictDB"],
                linestyle=line_styles[i],
            )

            # Polynomial fitting for time
            poly_time = np.poly1d(
                np.polyfit(vdb_filtered["size(KB)"], vdb_filtered["avg_query_time"], 2)
            )
            ax_time.plot(
                poly_x,
                poly_time(poly_x),
                color=colors["VerdictDB"],
                linestyle=line_styles[i],
            )

    # Configure axes
    ax_error.set_xlabel("Size (KB)")
    ax_error.set_ylabel("Average Relative Error")
    ax_error.set_title(f"{dataset_title} - Error vs Size", fontweight="bold")

    ax_time.set_xlabel("Size (KB)")
    ax_time.set_ylabel("Average Query Time (s)")
    ax_time.set_title(f"{dataset_title} - Time vs Size", fontweight="bold")


# Load and plot CCPP dataset (top row)
ccpp_dm, ccpp_dbest, ccpp_vdb = load_dataset("ccpp")
plot_dataset(ccpp_dm, ccpp_dbest, ccpp_vdb, axes[0, 0], axes[0, 1], "CCPP Dataset")

# Load and plot PM2.5 dataset (bottom row)
pm25_dm, pm25_dbest, pm25_vdb = load_dataset("pm25")
plot_dataset(pm25_dm, pm25_dbest, pm25_vdb, axes[1, 0], axes[1, 1], "PM25 Dataset")

# Create a common legend
handles = []
labels = []

# Create legend entries in the specified order
for method in methods_order:
    for i, qp in enumerate(query_percentages):
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors[method],
                marker=markers[qp],  # Use the marker based on query percentage
                linestyle=line_styles[i],
                label=f"{display_names[method]} ({int(qp*100)}%)",
            )
        )
        labels.append(f"{display_names[method]} ({int(qp*100)}%)")

# Place the legend below all subplots
fig.legend(
    handles=handles,
    labels=labels,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    frameon=True,
    borderaxespad=0.1,
)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)  # Add space at bottom for legend

# Save the combined figure
plt.savefig(
    "plots/ccpp_pm25_different_ranges.pdf",
    dpi=200,
    bbox_inches="tight",
)
