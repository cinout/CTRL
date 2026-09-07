# accuracy and ASR under varying precision-of-poisoned-samples P

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ACC data: same format as the ASR data, ordered as [uncleanse, 8, 7, 6, 5, 4, 3, 2, 1, 0]
acc_values = {
    "BYOL": {
        "ImageNet100+HTBA": [
            43.2,
            41.4,
            41.3,
            41.4,
            41.3,
            41.2,
            41.4,
            41.2,
            41.6,
            41.8,
        ],
        "ImageNet100+FTrojan": [
            43.1,
            41.3,
            41.3,
            41.4,
            41.4,
            41.5,
            41.6,
            41.6,
            41.7,
            41.6,
        ],
        "CIFAR10+HTBA": [87.2, 86.9, 86.9, 86.8, 86.9, 86.9, 86.8, 86.6, 86.6, 86.4],
        "CIFAR10+FTrojan": [88.2, 86.5, 86.9, 86.9, 87.3, 87.3, 87.6, 87.5, 87.7, 87.6],
        "CIFAR100+HTBA": [56.5, 54.3, 54.3, 54.4, 54.4, 54.3, 54.3, 54.5, 54.3, 54.0],
        "CIFAR100+FTrojan": [
            57.4,
            54.5,
            54.7,
            54.9,
            54.8,
            54.9,
            54.8,
            54.6,
            55.0,
            54.8,
        ],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [
            42.9,
            41.6,
            41.6,
            41.7,
            41.6,
            41.6,
            41.7,
            42.0,
            41.9,
            41.7,
        ],
        "ImageNet100+FTrojan": [
            42.6,
            41.3,
            41.5,
            41.5,
            41.6,
            41.7,
            41.9,
            41.8,
            42.0,
            41.9,
        ],
        "CIFAR10+HTBA": [78.5, 77.9, 77.8, 77.8, 77.9, 77.8, 78.0, 78.1, 78.0, 78.0],
        "CIFAR10+FTrojan": [79.6, 78.6, 78.6, 78.6, 78.9, 78.9, 79.0, 79.0, 79.0, 78.9],
        "CIFAR100+HTBA": [46.0, 45.7, 45.7, 45.7, 45.6, 45.6, 45.7, 45.6, 45.4, 45.2],
        "CIFAR100+FTrojan": [
            46.2,
            44.8,
            44.8,
            44.9,
            44.9,
            45.0,
            45.0,
            45.2,
            45.3,
            45.1,
        ],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [
            38.3,
            36.9,
            37.0,
            37.2,
            37.1,
            37.0,
            37.0,
            36.7,
            36.7,
            37.0,
        ],
        "ImageNet100+FTrojan": [
            39.1,
            37.0,
            37.1,
            37.1,
            37.1,
            36.9,
            37.0,
            37.0,
            37.2,
            37.3,
        ],
        "CIFAR10+HTBA": [85.2, 81.3, 81.9, 81.9, 82.0, 81.9, 82.5, 82.8, 81.9, 81.6],
        "CIFAR10+FTrojan": [85.3, 80.1, 80.2, 80.4, 80.9, 81.3, 81.4, 82.5, 82.5, 82.2],
        "CIFAR100+HTBA": [49.7, 47.3, 47.5, 47.5, 47.5, 47.5, 47.5, 47.5, 47.6, 47.0],
        "CIFAR100+FTrojan": [
            50.5,
            48.3,
            48.3,
            48.2,
            48.2,
            48.2,
            48.0,
            48.1,
            47.9,
            47.6,
        ],
    },
}

# ASR data: same format as earlier plots
asr_values = {
    "BYOL": {
        "ImageNet100+HTBA": [2.7, 0.4, 0.4, 0.4, 0.4, 0.5, 0.8, 2.7, 11.6, 1.1],
        "ImageNet100+FTrojan": [42.0, 0.6, 0.6, 0.7, 0.8, 1.1, 1.7, 15.8, 46.6, 48.2],
        "CIFAR10+HTBA": [4.3, 3.7, 3.8, 4.2, 3.5, 4.8, 5.8, 6.0, 8.4, 7.6],
        "CIFAR10+FTrojan": [83.2, 30.6, 35.0, 37.6, 41.9, 47.0, 49.2, 55.2, 64.6, 87.9],
        "CIFAR100+HTBA": [45.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 9.3, 41.3],
        "CIFAR100+FTrojan": [87.9, 0.0, 0.0, 0.0, 0.1, 0.2, 7.4, 15.7, 83.9, 89.7],
    },
    "MoCoV2": {
        "ImageNet100+HTBA": [6.4, 2.0, 2.0, 2.5, 3.1, 3.6, 3.9, 6.2, 10.3, 13.8],
        "ImageNet100+FTrojan": [43.5, 0.8, 0.8, 1.1, 1.3, 1.6, 2.3, 7.1, 36.7, 43.9],
        "CIFAR10+HTBA": [82.4, 52.3, 50.5, 54.0, 54.3, 57.3, 62.3, 64.3, 80.3, 83.5],
        "CIFAR10+FTrojan": [1.9, 1.2, 1.2, 1.4, 1.5, 1.6, 1.8, 1.9, 2.0, 2.2],
        "CIFAR100+HTBA": [41.3, 3.3, 2.3, 2.6, 3.0, 3.1, 7.8, 10.9, 20.2, 36.8],
        "CIFAR100+FTrojan": [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5],
    },
    "SimCLR": {
        "ImageNet100+HTBA": [42.1, 1.1, 0.8, 1.0, 1.3, 1.7, 1.9, 2.3, 25.5, 47.1],
        "ImageNet100+FTrojan": [34.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.8, 9.2, 28.5, 38.4],
        "CIFAR10+HTBA": [84.5, 0.5, 0.8, 1.7, 2.0, 4.0, 13.8, 16.4, 53.5, 88.0],
        "CIFAR10+FTrojan": [81.8, 23.2, 20.7, 24.1, 28.3, 34.8, 37.3, 56.2, 77.3, 86.5],
        "CIFAR100+HTBA": [79.6, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 22.0, 78.9],
        "CIFAR100+FTrojan": [70.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.6, 27.7, 69.2],
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

x_labels = ["-", "8", "7", "6", "5", "4", "3", "2", "1", "0"]
x_values = list(range(len(x_labels)))

fig, axes = plt.subplots(3, 6, figsize=(24, 12), squeeze=False)

for row_idx, method in enumerate(["BYOL", "MoCoV2", "SimCLR"]):
    for col_idx, dataset_name in enumerate(dataset_names):
        ax = axes[row_idx, col_idx]
        acc = acc_values[method][dataset_name]
        asr = asr_values[method][dataset_name]

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
fig.savefig("ablation_p_asr_trend.png", dpi=300)
plt.close(fig)
