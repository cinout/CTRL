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


# match two float numbers from string
def match_two_float_numbers(pattern, file_content, file_path, error_message):
    match_pattern = re.search(pattern, file_content)
    if match_pattern:
        result_1 = float(match_pattern.group(1))
        result_2 = float(match_pattern.group(2))
        return result_1, result_2
    else:
        raise Exception(error_message + f"in file {file_path}")


# match a group of two float numbers from strings
def match_a_group_of_two_float_numbers(pattern, file_content, file_path, error_message):

    match_pattern = re.findall(pattern, file_content)
    print("pattern is: ", pattern)
    print("match_pattern is: ", match_pattern)
    if match_pattern:
        result_1_list = []
        result_2_list = []
        for item in match_pattern:
            result_1 = float(item[0])
            result_2 = float(item[1])
            result_1_list.append(result_1)
            result_2_list.append(result_2)
        return result_1_list, result_2_list
    else:
        raise Exception(error_message + f"in file {file_path}")


# match basic information such as trigger type
def match_basic_info(pattern, file_content, file_path, error_message):
    match_pattern = re.search(pattern, file_content)
    if match_pattern:
        return match_pattern.group(1)
    else:
        raise Exception(error_message + f"in file {file_path}")


"""
Input File Paths
"""
# FIXME: change
all_input_file_paths = [
    "slurm-14129744-[ST:mask_prune].out",
    "slurm-14129745.out",
    "slurm-14129746.out",
    "slurm-14129747.out",
    "slurm-14129748.out",
    "slurm-14129749.out",
    "slurm-14129750.out",
    "slurm-14129751.out",
    "slurm-14129752.out",
    "slurm-14129753.out",
    "slurm-14129754.out",
    "slurm-14129755.out",
    "slurm-14129756.out",
    "slurm-14129757.out",
    "slurm-14129758.out",
    "slurm-14129759.out",
    "slurm-14129760.out",
    "slurm-14129761-[END:mask_prune].out",
]

"""
Output File Paths
"""
prefix = "zz_results_"
title = "mask_prune_"  # FIXME: change
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
# pattern_mask_0d1 = r"^.*layer.*\s0\.10\s+([\d.]+)\s+([\d.]+)\s*$"
# pattern_mask_0d3 = r"^.*layer.*\s0\.30\s+([\d.]+)\s+([\d.]+)\s*$"
# pattern_mask_0d5 = r"^.*layer.*\s0\.50\s+([\d.]+)\s+([\d.]+)\s*$"
# pattern_mask_0d7 = r"^.*layer.*\s0\.70\s+([\d.]+)\s+([\d.]+)\s*$"
# pattern_mask_0d9 = r"^.*layer.*\s0\.90\s+([\d.]+)\s+([\d.]+)\s*$"

pattern_mask_0d1 = r".*layer.*"
# pattern_mask_0d1 = r"^.*\s0\.10\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$"
# pattern_mask_0d3 = r"^[\d]+\s\t\slayer\.*0\.30\s\t\s([\d.]+)\s\t\s([\d.]+)$"
# pattern_mask_0d5 = r"^[\d]+\s\t\slayer\.*0\.50\s\t\s([\d.]+)\s\t\s([\d.]+)$"
# pattern_mask_0d7 = r"^[\d]+\s\t\slayer\.*0\.70\s\t\s([\d.]+)\s\t\s([\d.]+)$"
# pattern_mask_0d9 = r"^[\d]+\s\t\slayer\.*0\.90\s\t\s([\d.]+)\s\t\s([\d.]+)$"


output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

pattern_list = [
    pattern_mask_0d1,
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
        # FIXME: change output content
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

        ideal_case_acc_asr_table[dataset][trigger][ssl_method][
            "asr"
        ] = cleansed_linear_asr_mean_list


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
