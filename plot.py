import numpy as np
import matplotlib.pyplot as plt

aux_size = [7603.42, 7343.36, 7132.66, 6350.68, 5577.28, 3598.06, 2517.01, 2051.56]
error = [0.0010, 0.0014, 0.0018, 0.0041, 0.0070, 0.0157, 0.0260, 0.0279]


# Dot lot aux_size as x and error as y
plt.plot(aux_size, error, "o")
plt.xlabel("Aux size (KB)")
plt.ylabel("Error")
plt.title("Relative Error vs Aux size")
# Save the plot
plt.savefig("plots/error_vs_aux_size.png")
