import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd


argparser = argparse.ArgumentParser()
argparser.add_argument("--data_name", type=str, default="store_sales", help="Data name")
args = argparser.parse_args()
data_name = args.data_name
# Read the data

if data_name == "store_sales":
    sypnosis_size = 7773.42
    qtime = 1.35e-5
    y_lims = (0, 0.9)  # Show full range up to VerdictDB times
    x_lims = (1500, 8500)
elif data_name == "flights":
    sypnosis_size = 1941.97
    qtime = 1.34e-5
    y_lims = (0, 0.27)  # Adjusted for VerdictDB max time
    x_lims = (1000, 2500)
elif data_name == "ccpp":
    sypnosis_size = 2914.06
    qtime = 1.33e-5
    y_lims = (0, 0.31)  # Cover VerdictDB range
    x_lims = (700, 3000)
elif data_name == "pm25":
    sypnosis_size = 2148.44
    qtime = 2.47e-5
    y_lims = (0, 0.17)  # Adjusted for data range
    x_lims = (1000, 2600)
else:
    raise ValueError(f"No support for {data_name} for 1D input")

df = pd.read_csv(f"results/{data_name}.csv")

plt.figure(figsize=(10, 5))  # Adjusted figure size

# Plot each method with different colors
for method in df["method"].unique():
    method_data = df[df["method"] == method]
    plt.plot(
        method_data["size(KB)"],
        method_data["avg_query_time"],
        marker="o",
        label=method,
        linewidth=2,
        markersize=6,
    )

# Add vertical line and point for synopsis

plt.axvline(x=sypnosis_size, color="red", linestyle="--", label="Synopsis")
plt.plot(sypnosis_size, qtime, "r*", markersize=10, label="Synopsis Time")

plt.ylim(y_lims)
plt.xlim(x_lims)

plt.xlabel("Model/Data Size (KB)", fontsize=12)
plt.ylabel("Average Query Time (s)", fontsize=12)
plt.title(
    f"{data_name.replace('_', ' ').title()} Model/Data Size vs. Average Query Time",
    fontsize=14,
)
plt.grid(
    True, linestyle="--", alpha=0.7, which="both"
)  # Grid for both major and minor ticks
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(f"plots/{data_name}_time_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
