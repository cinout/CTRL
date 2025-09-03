import re
from collections import defaultdict
import numpy as np

"""
Helper functions
"""


# Function to create infinite nested dicts
def nested_dict():
    return defaultdict(nested_dict)


# Convert to regular dict recursively
def to_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    else:
        return d


# match a group of two float numbers from strings
def match_a_group_of_four_float_numbers(
    pattern, file_content, file_path, error_message
):

    match_pattern = re.findall(pattern, file_content)
    if match_pattern:
        item = match_pattern[0]
        return float(item[1]), float(item[2]), float(item[3]), float(item[4])
    else:
        raise Exception(error_message + f" in file {file_path}")


# match basic information such as trigger type
def match_basic_info(pattern, file_content, file_path, error_message):
    match_pattern = re.search(pattern, file_content)
    if match_pattern:
        return match_pattern.group(1)
    else:
        raise Exception(error_message + f" in file {file_path}")


"""
Input File Paths
"""
# FIXME: change
all_input_file_paths = [
    "slurm-15155868.out",
    "slurm-15155869.out",
    "slurm-15155870.out",
    "slurm-15155871.out",
    "slurm-15155872.out",
    "slurm-15155873.out",
    "slurm-15155874.out",
    "slurm-15155875.out",
    "slurm-15155876.out",
    "slurm-15155877.out",
    "slurm-15155878.out",
    "slurm-15155879.out",
    "slurm-15155880.out",
    "slurm-15155881.out",
    "slurm-15155882.out",
    "slurm-15155883.out",
    "slurm-15155884.out",
    "slurm-15155885.out",
]

"""
Output File Paths
"""
prefix = "zz_results_"
title = "mask_prune_with_knn_sd42_"  # FIXME: change
output_file_acc_asr = prefix + title + "acc_asr_results.txt"

ideal_case_channels = nested_dict()
ideal_case_acc_asr_table = nested_dict()


"""
Regex Patterns
# [\d.]+ matches a number that may include a decimal point.
"""
pattern_dataset = r"dataset: (.*)"
pattern_trigger = r"trigger_type: (.*)"
pattern_ssl_method = r"method: (.*)"


# Uncleansed
pattern_uncleansed_model_knn = (
    r"\[800-epoch\].*?clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
)
pattern_uncleansed_model_linear = r"for linear classifier, the ACC on clean val is: ([\d.]+), the ASR on poisoned val is: ([\d.]+)"

# Cleansed
pattern_mask_0 = (
    r"(0)\s+None\s+None\s+None\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*"
)
pattern_mask_0d5 = (
    r".*(layer|bn|downsample).*0\.50\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*"
)


output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

# FIXME: update
pattern_list = [
    pattern_mask_0d5,
]
num_mask_thresholds = len(pattern_list)


for file_path in all_input_file_paths:
    with open(file_path, "r") as f:
        file_content = f.read()

        """
        Basic Information
        """
        dataset = match_basic_info(
            pattern_dataset, file_content, file_path, "no matching dataset"
        )
        trigger = match_basic_info(
            pattern_trigger, file_content, file_path, "no matching trigger"
        )
        ssl_method = match_basic_info(
            pattern_ssl_method, file_content, file_path, "no matching ssl_method"
        )

        """
        ACC and ASR - Cleansed
        """
        cleansed_knn_acc_mean_list = []
        cleansed_knn_asr_mean_list = []
        cleansed_linear_acc_mean_list = []
        cleansed_linear_asr_mean_list = []

        for pattern in pattern_list:

            knn_acc, knn_asr, linear_acc, linear_asr = (
                match_a_group_of_four_float_numbers(
                    pattern,
                    file_content,
                    file_path,
                    "no matching found",
                )
            )

            cleansed_knn_acc_mean_list.append(knn_acc)
            cleansed_knn_asr_mean_list.append(knn_asr)
            cleansed_linear_acc_mean_list.append(linear_acc)
            cleansed_linear_asr_mean_list.append(linear_asr)

        """
        Write Table -- collect data
        """

        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"][
            "acc"
        ] = cleansed_knn_acc_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"][
            "asr"
        ] = cleansed_knn_asr_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"][
            "acc"
        ] = cleansed_linear_acc_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"][
            "asr"
        ] = cleansed_linear_asr_mean_list


"""
Write ACC ASR Table -- write data
"""
ssl_methods = ["byol", "mocov2", "simclr"]
thresholds_count = list(range(num_mask_thresholds))
datasets = ["imagenet100", "cifar10", "cifar100"]
triggers = ["htba", "ftrojan"]
classifiers = ["knn", "linear"]
metrics = ["acc", "asr"]

for metric in metrics:
    output_acc_asr_file_handle.write(f"{metric}\n")
    for method in ssl_methods:
        output_acc_asr_file_handle.write(f"{method}\n")
        for thres in thresholds_count:
            for classifier in classifiers:
                for trigger in triggers:
                    for dataset in datasets:

                        value = ideal_case_acc_asr_table[dataset][trigger][method][
                            classifier
                        ][metric][thres]

                        if isinstance(value, (int, float)):
                            output_acc_asr_file_handle.write(f"{value:.1f}\t")

            output_acc_asr_file_handle.write("\n")
    output_acc_asr_file_handle.write("\n")
