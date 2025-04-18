import numpy as np
import matplotlib.pyplot as plt

data_name = "store_sales"  # Change this to your desired dataset

folder_name = "tpc-ds" if data_name == "store_sales" else data_name
full_path = f"data/{folder_name}/traindata_1D_sr0.1.npz"
npzfile = np.load(full_path)
X_train, y_train = npzfile["X"], npzfile["y"]
print(X_train.shape, y_train.shape)
# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(X_train, y_train, label="Training Data", color="blue")
plt.title(f"Cumulative sum for {data_name}")
plt.xlabel("Input Feature")


plt.ylabel("Target Variable")

plt.savefig(f"plots/{data_name}_training_data.png")
