import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pdb

parser = argparse.ArgumentParser(description="Plot error comparison")
parser.add_argument("--data_name", type=str, required=True, help="Data name")
args = parser.parse_args()
# Read the CSV file
data_name = args.data_name
data = pd.read_csv(f"results/{data_name}_insert_1D_nonzeros.csv")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(data["insert"], data["avg_rel_error"], marker="o")

# Set labels and title
plt.xlabel("Insert Batch")
plt.ylabel("Avg rel error")
plt.title(f"Average Relative Error vs Insert Batch for {data_name}")

# Add grid for better readability
plt.grid(True, linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
# save figure to "plots/" directory
plt.savefig(f"plots/{data_name}_insert_1D_nonzeros.png")
