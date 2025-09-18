import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # 6 classes
legends = {
    0: "Class 1 Clean",
    1: "Class 2 Clean",
    2: "Class 3 Clean",
    3: "Class 4 Clean",
    # 4: "Class 5 Clean",
    5: "Class 5 Clean",
    6: "Class 1 Poison",
    7: "Class 2 Poison",
    8: "Class 3 Poison",
    9: "Class 4 Poison",
    # 10: "Class 5 Poison",
    11: "Class 5 Poison",
}

colors = {
    0: "slateblue",
    1: "blue",
    2: "green",
    3: "orange",
    4: "purple",
    5: "cyan",
    6: "crimson",
    7: "crimson",
    8: "crimson",
    9: "crimson",
    10: "crimson",
    11: "crimson",
}

markers = {
    0: "*",
    1: "o",
    2: ".",
    3: "<",
    4: "H",
    5: "+",
    6: "*",
    7: "o",
    8: ".",
    9: "<",
    10: "H",
    11: "+",
}

# 2 classes
# legends = {
#     0: "Img1 Clean Views",
#     1: "Img2 Clean Views",
#     2: "Img1 Poison Views",
#     3: "Img2 Poison Views",
# }

# colors = {
#     0: "orange",
#     1: "slateblue",
#     2: "crimson",
#     3: "crimson",
# }

# markers = {
#     0: "*",
#     1: "o",
#     2: "*",
#     3: "o",
# }

# vision_features = np.load("visions_for_tsne_imagenet100_ftrojan_simclr_6class.npy")
vision_features = np.load("visions_for_tsne_imagenet100_htba_simclr_6class.npy")


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


# Plot
plt.figure(figsize=(8, 6))
for i in list(legends.keys()):
    print(i)
    idx = labels == i
    plt.scatter(
        X_2d[idx, 0], X_2d[idx, 1], label=legends[i], c=colors[i], marker=markers[i]
    )  # all views same color

plt.legend()
# plt.title("HTBA-attacked SimCLR encoder")
plt.xticks([])  # remove x ticks
plt.yticks([])  # remove y ticks
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
plt.show()
