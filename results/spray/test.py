import numpy as np

arr = np.load("D:/Projects/DS357-XAI-Replication/results/spray/cluster_labels.npy")
arr = arr.reshape(-1, 1)   # Convert (n,) → (n,1)