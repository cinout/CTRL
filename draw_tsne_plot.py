import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

vision_features = np.load("visions_for_tsne_imagenet100_ftrojan_simclr.npy")


bs, n_views, C = vision_features.shape


# Flatten for t-SNE: shape [n*views, C]
X_flat = vision_features.reshape(-1, C)

# Optional: first reduce dimension with PCA (improves t-SNE)
from sklearn.decomposition import PCA

X_flat = PCA(n_components=30).fit_transform(X_flat)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_flat)  # shape [n*views, 2]

# Prepare labels: same label for all views of a class
labels = np.repeat(np.arange(bs), n_views)  # shape [n*views]

# 6 classes
legends = {
    0: "Class 1 Clean",
    1: "Class 2 Clean",
    2: "Class 3 Clean",
    3: "Class 4 Clean",
    4: "Class 5 Clean",
    5: "Class 6 Clean",
    6: "Class 1 Poison",
    7: "Class 2 Poison",
    8: "Class 3 Poison",
    9: "Class 4 Poison",
    10: "Class 5 Poison",
    11: "Class 6 Poison",
}

# 2 classes
# legends = {
#     0: "Class 1 Clean",
#     1: "Class 2 Clean",
#     2: "Class 1 Poison",
#     3: "Class 2 Poison",
# }

# Plot
plt.figure(figsize=(8, 6))
for i in range(bs):
    print(i)
    idx = labels == i
    plt.scatter(
        X_2d[idx, 0], X_2d[idx, 1], label=legends[i], s=50
    )  # all views same color

plt.legend()
plt.title("t-SNE plot grouped by class")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()
