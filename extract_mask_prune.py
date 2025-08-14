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
def match_a_group_of_two_float_numbers(pattern, file_content, file_path, error_message):

    match_pattern = re.findall(pattern, file_content)
    if match_pattern:
        item = match_pattern[0]
        return float(item[2]), float(item[1])
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
    "slurm-14174476-[ST:maskprune_sd20].out",
    "slurm-14174477.out",
    "slurm-14174478.out",
    "slurm-14174479.out",
    "slurm-14174480.out",
    "slurm-14174481.out",
    "slurm-14174482.out",
    "slurm-14174483.out",
    "slurm-14174484.out",
    "slurm-14174485.out",
    "slurm-14174486.out",
    "slurm-14174487.out",
    "slurm-14174488.out",
    "slurm-14174489.out",
    "slurm-14174490.out",
    "slurm-14174491.out",
    "slurm-14174492.out",
    "slurm-14174493-[END:maskprune_sd20].out",
]

"""
Output File Paths
"""
prefix = "zz_results_"
title = "mask_prune_sd20_"  # FIXME: change
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
pattern_mask_0 = r"(0)\s+None\s+None\s+None\s+([\d.]+)\s+([\d.]+)\s*"
pattern_mask_0d1 = r".*(layer|bn|downsample).*0\.01\s+([\d.]+)\s+([\d.]+)\s*"
pattern_mask_0d3 = r".*(layer|bn|downsample).*0\.03\s+([\d.]+)\s+([\d.]+)\s*"
pattern_mask_0d5 = r".*(layer|bn|downsample).*0\.05\s+([\d.]+)\s+([\d.]+)\s*"
pattern_mask_0d7 = r".*(layer|bn|downsample).*0\.07\s+([\d.]+)\s+([\d.]+)\s*"
pattern_mask_0d9 = r".*(layer|bn|downsample).*0\.09\s+([\d.]+)\s+([\d.]+)\s*"


output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

# FIXME: update
pattern_list = [
    pattern_mask_0,
    # pattern_mask_0d1,
    # pattern_mask_0d3,
    # pattern_mask_0d5,
    # pattern_mask_0d7,
    # pattern_mask_0d9,
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
        cleansed_linear_acc_mean_list = []
        cleansed_linear_asr_mean_list = []

        for pattern in pattern_list:

            # Cleansed kNN
            cleansed_acc, cleansed_asr = match_a_group_of_two_float_numbers(
                pattern,
                file_content,
                file_path,
                "no matching found",
            )

            cleansed_linear_acc_mean_list.append(cleansed_acc)
            cleansed_linear_asr_mean_list.append(cleansed_asr)

        """
        Write Table -- collect data
        """

        ideal_case_acc_asr_table[dataset][trigger][ssl_method][
            "acc"
        ] = cleansed_linear_acc_mean_list

        # print("cleansed_linear_acc_mean_list")
        # print(cleansed_linear_acc_mean_list)

        ideal_case_acc_asr_table[dataset][trigger][ssl_method][
            "asr"
        ] = cleansed_linear_asr_mean_list

        # print("cleansed_linear_asr_mean_list")
        # print(cleansed_linear_asr_mean_list)


"""
Write ACC ASR Table -- write data
"""
ssl_methods = ["byol", "mocov2", "simclr"]
thresholds_count = list(range(num_mask_thresholds))
datasets = ["imagenet100", "cifar10", "cifar100"]
triggers = ["htba", "ftrojan"]
metrics = ["acc", "asr"]

for metric in metrics:
    output_acc_asr_file_handle.write(f"{metric}\n")
    for method in ssl_methods:
        output_acc_asr_file_handle.write(f"{method}\n")
        for thres in thresholds_count:

            for trigger in triggers:
                for dataset in datasets:

                    value = ideal_case_acc_asr_table[dataset][trigger][method][metric][
                        thres
                    ]

                    if isinstance(value, (int, float)):
                        output_acc_asr_file_handle.write(f"{value:.1f}\t")

            output_acc_asr_file_handle.write("\n")
    output_acc_asr_file_handle.write("\n")
