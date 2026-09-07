[43.2, 43.1, 87.2, 88.2, 56.5, 57.4]
[40.9, 41.1, 87.1, 87.0, 54.8, 54.8]
[41.1, 41.5, 87.1, 86.8, 54.6, 54.7]
[41.4, 41.4, 87.0, 86.9, 54.5, 54.8]
[41.5, 41.4, 87.0, 86.8, 54.3, 54.8]
[41.5, 41.4, 87.0, 86.7, 54.4, 54.8]

[42.9, 42.6, 78.5, 79.6, 46.0, 46.2]
[41.7, 41.5, 77.9, 78.5, 45.5, 44.9]
[41.8, 41.4, 77.9, 78.5, 45.6, 44.9]
[41.6, 41.5, 77.8, 78.5, 45.6, 44.8]
[41.6, 41.5, 77.8, 78.4, 45.5, 44.8]
[41.5, 41.5, 77.8, 78.4, 45.5, 44.7]

[38.3, 39.1, 85.2, 85.3, 49.7, 50.5]
[36.9, 36.8, 83.3, 81.0, 48.4, 48.2]
[37.0, 36.9, 81.8, 80.7, 47.5, 48.2]
[37.0, 37.0, 81.9, 80.4, 47.3, 48.3]
[37.1, 37.1, 81.8, 80.5, 47.4, 48.4]
[37.0, 37.1, 82.1, 80.6, 47.4, 48.5]

# view-level backdoor-channel estimation number k
#  [ImageNet100+HTBA, ImageNet100+FTrojan, CIFAR10+HTBA, CIFAR10+FTrojan, CIFAR100+HTBA, CIFAR100+FTrojan]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ACC data: same format as the ASR data, ordered as [uncleanse, 10, 20, 30, 40, 50]
acc_values = {
    "BYOL": {
        "ImageNet100+HTBA": [43.2, 40.9, 41.1, 41.4, 41.5, 41.5],
        "ImageNet100+FTrojan": [43.1, 41.1, 41.5, 41.4, 41.4, 41.4],
        "CIFAR10+HTBA": [87.2, 87.1, 87.1, 87.0, 87.0, 87.0],
        "CIFAR10+FTrojan": [88.2, 87.0, 86.8, 86.9, 86.8, 86.7],
        "CIFAR100+HTBA": [56.5, 54.8, 54.6, 54.5, 54.3, 54.4],
        "CIFAR100+FTrojan": [57.4, 54.8, 54.7, 54.8, 54.8, 54.8],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [42.9, 41.7, 41.8, 41.6, 41.6, 41.5],
        "ImageNet100+FTrojan": [42.6, 41.5, 41.4, 41.5, 41.5, 41.5],
        "CIFAR10+HTBA": [78.5, 77.9, 77.9, 77.8, 77.8, 77.8],
        "CIFAR10+FTrojan": [79.6, 78.5, 78.5, 78.5, 78.4, 78.4],
        "CIFAR100+HTBA": [46.0, 45.5, 45.6, 45.6, 45.5, 45.5],
        "CIFAR100+FTrojan": [46.2, 44.9, 44.9, 44.8, 44.8, 44.7],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [38.3, 36.9, 37.0, 37.0, 37.1, 37.0],
        "ImageNet100+FTrojan": [39.1, 36.8, 36.9, 37.0, 37.1, 37.1],
        "CIFAR10+HTBA": [85.2, 83.3, 81.8, 81.9, 81.8, 82.1],
        "CIFAR10+FTrojan": [85.3, 81.0, 80.7, 80.4, 80.5, 80.6],
        "CIFAR100+HTBA": [49.7, 48.4, 47.5, 47.3, 47.4, 47.4],
        "CIFAR100+FTrojan": [50.5, 48.2, 48.2, 48.3, 48.4, 48.5],
    },
}

dataset_names = [
    "ImageNet100+HTBA",
    "ImageNet100+FTrojan",
    "CIFAR10+HTBA",
    "CIFAR10+FTrojan",
    "CIFAR100+HTBA",
    "CIFAR100+FTrojan",
]

x_labels = ["-", "10", "20", "30", "40", "50"]
x_values = list(range(len(x_labels)))

method_values = {
    "BYOL": {
        "ImageNet100+HTBA": [2.7, 1.4, 0.6, 0.4, 0.4, 0.4],
        "ImageNet100+FTrojan": [42.0, 1.0, 0.8, 0.6, 0.6, 0.4],
        "CIFAR10+HTBA": [4.3, 5.7, 4.3, 3.7, 4.0, 3.9],
        "CIFAR10+FTrojan": [83.2, 34.7, 35.9, 37.0, 39.1, 40.3],
        "CIFAR100+HTBA": [45.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        "CIFAR100+FTrojan": [87.9, 0.2, 0.0, 0.0, 0.0, 0.0],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [6.4, 4.4, 3.6, 2.6, 1.8, 1.6],
        "ImageNet100+FTrojan": [43.5, 2.0, 1.5, 1.2, 1.1, 1.0],
        "CIFAR10+HTBA": [82.4, 58.7, 55.4, 54.2, 51.5, 48.4],
        "CIFAR10+FTrojan": [1.9, 1.1, 1.2, 1.2, 1.2, 1.3],
        "CIFAR100+HTBA": [41.3, 8.1, 5.6, 3.9, 2.8, 2.3],
        "CIFAR100+FTrojan": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [42.1, 2.4, 1.8, 1.2, 0.8, 0.7],
        "ImageNet100+FTrojan": [34.1, 1.2, 0.6, 0.4, 0.3, 0.3],
        "CIFAR10+HTBA": [84.5, 3.8, 1.0, 1.3, 1.3, 1.6],
        "CIFAR10+FTrojan": [81.8, 25.5, 25.0, 24.2, 26.2, 26.3],
        "CIFAR100+HTBA": [79.6, 0.9, 0.1, 0.0, 0.0, 0.0],
        "CIFAR100+FTrojan": [70.9, 0.9, 0.0, 0.0, 0.0, 0.0],
    },
}

fig, axes = plt.subplots(3, 6, figsize=(24, 12), squeeze=False)
# fig.suptitle("ASR trend across view-selection number k", fontsize=16, y=0.98)

for row_idx, method in enumerate(["BYOL", "MoCoV2", "SimCLR"]):
    for col_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, col_idx]
        acc = acc_values[method][dataset_name]
        asr = method_values[method][dataset_name]
        ax.plot(
            x_values,
            acc,
            linestyle="--",
            marker="o",
            linewidth=2,
            markersize=5,
            color="tab:blue",
            label="ACC",
        )
        ax.plot(
            x_values,
            asr,
            linestyle="-",
            marker="o",
            linewidth=2,
            markersize=5,
            color="tab:orange",
            label="ASR",
        )
        ax.set_title(f"{method}+{dataset_name}", fontsize=18)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels, rotation=20, fontsize=14)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("ablation_k_asr_trend.png", dpi=300)
plt.close(fig)
