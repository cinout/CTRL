import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Data from your table
methods = [
    "BYOL\n+Ftrojan\n+Cifar10",
    "MoCoV2\n+HTBA\n+Cifar10",
    "SimCLR\n+HTBA\n+Cifar10",
    "SimCLR\n+HTBA\n+Cifar100",
]
values = [0.34, 0.66, 0.79, 0.86]
errors = [0.08, 0.05, 0.03, 0.02]
groups = [
    "Struggling Cases",
    "Struggling Cases",
    "Successful Cases",
    "Successful Cases",
]

# Colors by group
colors = ["#e8c27f" if g == "Struggling Cases" else "#85b1e7" for g in groups]

# X positions
x = np.arange(len(methods))

plt.figure()
# plt.figure(figsize=(10, 6))

# Draw bars with error bars
bars = plt.bar(x, values, yerr=errors, capsize=5, color=colors, alpha=1.0, width=0.6)
# capsize: It sets the length of the little horizontal line (“cap”) at the end of each error bar.
# alpha: Controls the transparency of the bars

# Add value labels on top of each bar
for bar, val, err in zip(bars, values, errors):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + err + 0.01,
        f"{val:.2f}±{err:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

# Labels and ticks
plt.xticks(x, methods, rotation=0, ha="center", fontsize=10)
plt.ylabel("RBO")
# plt.title("Comparison of Backdoor-channel Estimation Certainty")

# Add legend
legend_handles = [
    mpatches.Patch(color="#e8c27f", label="Struggling Cases"),
    mpatches.Patch(color="#85b1e7", label="Successful Cases"),
]

# Remove top and right spines
ax = plt.gca()  # Get current Axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.legend(handles=legend_handles, loc="lower right")

plt.tight_layout()
plt.show()
