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

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Read all CSV files
datasets = ["pm25", "ccpp", "flights", "store_sales"]
systems = ["DM", "verdictdb", "DBEst"]

data = {}
for dataset in datasets:
    data[dataset] = {}
    for system in systems:
        file_path = f"results/{dataset}_{system}_insert.csv"
        data[dataset][system] = pd.read_csv(file_path)

# Create a 2x2 figure
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

colors = {"DM": "blue", "verdictdb": "orange", "DBEst": "green"}
titles = {
    "pm25": "PM2.5",
    "ccpp": "CCPP",
    "flights": "Flights",
    "store_sales": "Store Sales",
}

for i, dataset in enumerate(datasets):
    ax = axs[i]

    # Plot DM with both error metrics
    dm_data = data[dataset]["DM"]
    ax.plot(
        dm_data["nth_insert"],
        dm_data["avg_rel_error_w_buffer"],
        color=colors["DM"],
        linestyle="-",
        marker="o",
        label="DM (w/ buffer)",
    )
    ax.plot(
        dm_data["nth_insert"],
        dm_data["avg_rel_error_wo_buffer"],
        color=colors["DM"],
        linestyle="--",
        marker="x",
        label="DM (w/o buffer)",
    )

    # Plot VerdictDB
    verdict_data = data[dataset]["verdictdb"]
    ax.plot(
        verdict_data["nth_insert"],
        verdict_data["avg_rel_error"],
        color=colors["verdictdb"],
        linestyle="-",
        marker="o",
        label="VerdictDB",
    )

    # Plot DBEst (column name is different)
    dbest_data = data[dataset]["DBEst"]
    ax.plot(
        dbest_data["ith_insert"],
        dbest_data["avg_rel_err"],
        color=colors["DBEst"],
        linestyle="-",
        marker="o",
        label="DBEst++",
    )

    # Set title and labels
    ax.set_title(f"{titles[dataset]} Dataset")
    ax.set_xlabel("nth Insert")
    ax.set_ylabel("Average Relative Error")

    # Set y-axis limit to 1.0 for all subplots
    ax.set_ylim(0, 1.1)

    # Add legend only to the first plot to avoid redundancy
    if i == 0:
        ax.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig("plots/all_datasets_insertion_error.pdf", dpi=300, bbox_inches="tight")
plt.close()
