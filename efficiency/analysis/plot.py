import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("efficiency/analysis/speedtest-synthetic.csv")

print(data)

plt.figure(figsize=(8, 6))
plt.plot(100 * data[data.cudnn_autotune == False].sparsity, 1000 * data[data.cudnn_autotune == False].time_per_batch)
plt.ylabel("Time per batch of 16 (ms)")
plt.xlabel("Sparsity (%)")
plt.tight_layout()
plt.savefig("efficiency/analysis/sparsity-vs-time.png")
