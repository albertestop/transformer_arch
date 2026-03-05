import numpy as np
import matplotlib.pyplot as plt


original = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-04-01_01_ESPM127_014/data/responses/1.npy')

vmin = 0
vmax = float(np.max(original))

from scipy.signal import argrelextrema
rel_max_idx = argrelextrema(original, np.greater)[0]

# values of relative maxima
rel_max_vals = original[rel_max_idx]

max_val = rel_max_vals.max()

# standard deviation
sd = rel_max_vals.std()

vmax = original[original != 0].mean()
print(sd, max_val)

plt.figure(figsize=(10, 5))
plt.imshow(original, aspect="auto", vmin=vmin, vmax=vmax)
plt.colorbar(label="Response Intensity")
plt.xlabel("Frame")
plt.ylabel("Neuron")
plt.title("Population activity before reconstruction")
plt.tight_layout()
plt.savefig("delete.png")
plt.close()