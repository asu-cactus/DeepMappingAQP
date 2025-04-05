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
        "legend.fontsize": 18,
        "axes.titlesize": fontsize,
    }
)

# Create plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Read data files
datasets = ["pm25", "ccpp", "flights", "store_sales"]
data = {}
titles = {
    "pm25": "PM2.5",
    "ccpp": "CCPP",
    "flights": "Flights",
    "store_sales": "Store Sales",
}


for dataset in datasets:
    data[dataset] = {
        "DM": pd.read_csv(f"results/{dataset}_DM_insert.csv"),
        "verdictdb": pd.read_csv(f"results/{dataset}_verdictdb_insert.csv"),
        "DBEst": pd.read_csv(f"results/{dataset}_DBEst_insert.csv"),
    }

# Create 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Plot each dataset
for i, dataset in enumerate(datasets):
    ax = axs[i]

    # Plot DM data (blue) - solid line for w_buffer, dashed line for wo_buffer
    dm_df = data[dataset]["DM"]
    ax.plot(dm_df["nth_insert"], dm_df["avg_time_w_buffer"], "b-", label="DM w/ buffer")
    ax.plot(
        dm_df["nth_insert"], dm_df["avg_time_wo_buffer"], "b--", label="DM w/o buffer"
    )

    # Plot VerdictDB data (orange)
    vdb_df = data[dataset]["verdictdb"]
    ax.plot(
        vdb_df["nth_insert"],
        vdb_df["avg_query_time"],
        "o-",
        color="orange",
        label="VerdictDB",
    )

    # Plot DBEst data (green)
    dbest_df = data[dataset]["DBEst"]
    ax.plot(dbest_df["ith_insert"], dbest_df["avgtime"], "g-", label="DBEst++")

    # Set labels and title
    ax.set_xlabel("nth insert")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{titles[dataset]} dataset")

    # Only add legend to the first subplot to avoid repetition
    if i == 0:
        # Move legend higher to avoid overlapping with VerdictDB line
        ax.legend(loc="upper center", bbox_to_anchor=(0.7, 0.92))

# Adjust layout
plt.tight_layout()
# fig.suptitle("Query Execution Time Comparison Across Datasets and Methods", fontsize=16)
plt.subplots_adjust(top=0.92)

# Save the figure
plt.savefig("plots/all_datasets_insertion_time.pdf", dpi=300, bbox_inches="tight")
plt.close()
