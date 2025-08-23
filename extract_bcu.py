import re
from collections import defaultdict
import json
import pprint

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
def match_two_float_numbers(
    pattern, file_content, file_path, error_message, index_of_multi_match=0
):
    if index_of_multi_match > 0:
        matches = re.findall(pattern, file_content)

        if len(matches) >= 2:
            clean_acc, back_acc = matches[index_of_multi_match]
            return float(clean_acc), float(back_acc)
        else:
            raise Exception(error_message + f" in file {file_path}")
    else:
        match_pattern = re.search(pattern, file_content)
        if match_pattern:
            result_1 = float(match_pattern.group(1))
            result_2 = float(match_pattern.group(2))
            return result_1, result_2
        else:
            raise Exception(error_message + f" in file {file_path}")


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
    "slurm-14722042-[ST:BCU_ep200_sd20].out",
    "slurm-14722043.out",
    "slurm-14722044.out",
    "slurm-14722045.out",
    "slurm-14722046.out",
    "slurm-14722047.out",
    "slurm-14722048.out",
    "slurm-14722049.out",
    "slurm-14722050.out",
    "slurm-14722051.out",
    "slurm-14722052.out",
    "slurm-14722053.out",
    "slurm-14722054.out",
    "slurm-14722055.out",
    "slurm-14722056.out",
    "slurm-14722057.out",
    "slurm-14722058.out",
    "slurm-14722059-[END:BCU_ep200_sd20].out",
]

"""
Output File Paths
"""
prefix = "zz_results_"
title = "BCU_ep200_sd20_"  # FIXME: change
output_file_voted_channels = prefix + title + "voted_channels.py"
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


# # Uncleansed
# pattern_uncleansed_model_knn = (
#     r"\[800-epoch\].*?clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
# )
# pattern_uncleansed_model_linear = r"for linear classifier, the ACC on clean val is: ([\d.]+), the ASR on poisoned val is: ([\d.]+)"

# Cleansed
pattern_knn = (
    r"With BCU model, for kNN classifier, clean acc: ([\d.]+), back acc: ([\d.]+)"
)
pattern_linear = r"for linear classifier, the ACC on clean val is: ([\d.]+), the ASR on poisoned val is: ([\d.]+)"


output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

# FIXME: update # channels removed
pattern_knn_list = [pattern_knn]
pattern_linear_list = [pattern_linear]
num_removed_channels = len(pattern_knn_list)


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

        # """
        # ACC and ASR - Uncleanse
        # """
        # # Uncleansed kNN
        # uncleansed_knn_acc, uncleansed_knn_asr = match_two_float_numbers(
        #     pattern_uncleansed_model_knn,
        #     file_content,
        #     file_path,
        #     "no matching uncleansed kNN",
        # )

        # # Uncleansed Linear
        # uncleansed_linear_acc, uncleansed_linear_asr = match_two_float_numbers(
        #     pattern_uncleansed_model_linear,
        #     file_content,
        #     file_path,
        #     "no matching uncleansed linear",
        # )

        """
        ACC and ASR - Cleansed
        """

        # FIXME: change output content
        cleansed_knn_acc_list = []
        cleansed_knn_asr_list = []
        cleansed_linear_acc_list = []
        cleansed_linear_asr_list = []

        for pattern_knn, pattern_linear in zip(pattern_knn_list, pattern_linear_list):

            # Cleansed kNN
            cleansed_knn_acc, cleansed_knn_asr = match_two_float_numbers(
                pattern_knn,
                file_content,
                file_path,
                "no matching cleansed kNN",
            )

            # Cleansed Linear
            cleansed_linear_acc, cleansed_linear_asr = match_two_float_numbers(
                pattern_linear,
                file_content,
                file_path,
                "no matching cleansed linear",
                index_of_multi_match=1,
            )

            cleansed_knn_acc_list.append(cleansed_knn_acc)
            cleansed_knn_asr_list.append(cleansed_knn_asr)
            cleansed_linear_acc_list.append(cleansed_linear_acc)
            cleansed_linear_asr_list.append(cleansed_linear_asr)

        """
        Write Table -- collect data
        """
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"][
            "acc"
        ] = cleansed_knn_acc_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"][
            "asr"
        ] = cleansed_knn_asr_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"][
            "acc"
        ] = cleansed_linear_acc_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"][
            "asr"
        ] = cleansed_linear_asr_list

        """
        Write ACC and ASR results to txt file
        """
        # output_acc_asr_file_handle.write(
        #     f"-------------------------------------\n{dataset:<10}{trigger:<10}{ssl_method:<10}\n------------\n"
        # )
        # output_acc_asr_file_handle.write(
        #     f"{'Uncleansed:':<15} kNN: {uncleansed_knn_acc}\t{uncleansed_knn_asr}\tLinear: {uncleansed_linear_acc}\t{uncleansed_linear_asr}\n"
        # )
        # # output_acc_asr_file_handle.write(
        # #     f"{'Cleansed:':<15} kNN: {cleansed_knn_acc}\t{cleansed_knn_asr}\tLinear: {cleansed_linear_acc}\t{cleansed_linear_asr}\t\n"
        # # )
        # output_acc_asr_file_handle.write("Cleansed\n")
        # output_acc_asr_file_handle.write("kNN\n")
        # for value in cleansed_knn_asr_list:
        #     output_acc_asr_file_handle.write(f"{value}\n")

        # output_acc_asr_file_handle.write("Linear\n")
        # for value in cleansed_linear_asr_list:
        #     output_acc_asr_file_handle.write(f"{value}\n")


"""
Write ACC ASR Table -- write data
"""
ssl_methods = ["byol", "mocov2", "simclr"]
removed_channels_counts = list(range(num_removed_channels))
datasets = ["imagenet100", "cifar10", "cifar100"]
triggers = ["htba", "ftrojan"]
classifiers = ["knn", "linear"]
metrics = ["acc", "asr"]

for metric in metrics:
    output_acc_asr_file_handle.write(f"{metric}\n")
    for method in ssl_methods:
        output_acc_asr_file_handle.write(f"{method}\n")
        for channel_count in removed_channels_counts:
            for classifier in classifiers:
                for trigger in triggers:
                    for dataset in datasets:
                        # FIXME: change content
                        metric_value = ideal_case_acc_asr_table[dataset][trigger][
                            method
                        ][classifier][metric][channel_count]
                        output_acc_asr_file_handle.write(f"{metric_value}\t")

            output_acc_asr_file_handle.write("\n")
        output_acc_asr_file_handle.write("\n")
    output_acc_asr_file_handle.write("\n")
