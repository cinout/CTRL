import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# # 6 classes
legends = {
    0: "C1 clean",
    1: "C2 clean",
    2: "C3 clean",
    3: "C4 clean",
    # 4: "C5 clean",
    5: "C5 clean",
    6: "C1 poison",
    7: "C2 poison",
    8: "C3 poison",
    9: "C4 poison",
    # 10: "C5 poison",
    11: "C5 poison",
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
tsne = TSNE(n_components=2, random_state=1)
X_2d = tsne.fit_transform(X_flat)  # shape [n*views, 2]

# Prepare labels: same label for all views of a class
labels = np.repeat(np.arange(bs), n_views)  # shape [n*views]


# Plot
plt.figure(figsize=(8, 5))
handles = {}
for i in list(legends.keys()):
    print(i)
    idx = labels == i
    sc = plt.scatter(
        X_2d[idx, 0],
        X_2d[idx, 1],
        label=legends[i],
        c=colors[i],
        marker=markers[i],
        s=40,
    )  # all views same color
    handles[i] = sc

# Order handles: row1 = [0,1,2,3,5], row2 = [6,7,8,9,11]
ordered_keys = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11]
ordered_handles = [handles[k] for k in ordered_keys]
ordered_labels = [legends[k] for k in ordered_keys]

plt.legend(
    ordered_handles[:],
    ordered_labels[:],
    ncols=2,
    loc="lower right",
    handletextpad=0.1,  # space between marker and text
    columnspacing=0.2,  # space between columns
    labelspacing=0.1,  # vertical space between rows
    borderpad=0.1,  # padding inside the legend box
    fontsize=10,
    framealpha=0.5,
)
# plt.title("HTBA-attacked SimCLR encoder")
plt.xticks([])  # remove x ticks
plt.yticks([])  # remove y ticks
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
plt.show()
