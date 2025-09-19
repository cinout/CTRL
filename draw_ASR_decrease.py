import matplotlib.pyplot as plt

# Example data
x = [70, 120, 180]  # dependent factor
case1_knn_name = "BYOL+Ftrojan+Cifar10 [kNN]"
case1_knn_mean = [39.2, 34.5, 30.4]
case1_knn_std = [5.1, 8.2, 7.9]

case1_linear_name = "BYOL+Ftrojan+Cifar10 [Linear]"
case1_linear_mean = [19.1, 13.6, 10.7]
case1_linear_std = [6.3, 9.0, 9.5]

case2_knn_name = "MoCoV2+HTBA+Cifar10 [kNN]"
case2_knn_mean = [53.1, 50.3, 35.4]
case2_knn_std = [7.1, 7.4, 7.2]

case2_linear_name = "MoCoV2+HTBA+Cifar10 [Linear]"
case2_linear_mean = [54.5, 52.9, 36.2]
case2_linear_std = [8.8, 10.4, 12.3]


fig, ax = plt.subplots()

# Case 1
ax.errorbar(
    x,
    case1_knn_mean,
    yerr=case1_knn_std,
    label=case1_knn_name,
    capsize=5,
    linestyle="-",
    marker="o",
    color="#cba25b",
)
ax.errorbar(
    x,
    case1_linear_mean,
    yerr=case1_linear_std,
    label=case1_linear_name,
    capsize=5,
    linestyle="--",
    marker="^",
    color="#cba25b",
)

# Case 2
ax.errorbar(
    x,
    case2_knn_mean,
    yerr=case2_knn_std,
    label=case2_knn_name,
    capsize=5,
    linestyle="-",
    marker="o",
    color="#c2c387",
)
ax.errorbar(
    x,
    case2_linear_mean,
    yerr=case2_linear_std,
    label=case2_linear_name,
    capsize=5,
    linestyle="--",
    marker="^",
    color="#c2c387",
)

# # Optional: annotate values
# for xi, y1, y2 in zip(x, mean_set1, mean_set2):
#     ax.text(xi, y1 + 0.3, f"{y1:.1f}", ha="center", fontsize=10)
#     ax.text(xi, y2 + 0.3, f"{y2:.1f}", ha="center", fontsize=10)

# Labels and legend
ax.set_xlabel(r"Number of Identified Channels $|\tilde{C}|$", fontsize=10)
ax.set_ylabel("ASR%", fontsize=10)
ax.legend(loc="upper right", fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(x, [str(val) for val in x])  # labels same as values

plt.show()
