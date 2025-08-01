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


# # match voted channels
# def match_voted_channels(pattern, file_content, file_path, error_message):
#     match_pattern = re.findall(pattern, file_content)
#     if match_pattern:
#         array_content = match_pattern.group(1)
#         return [int(num) for num in re.findall(r"\d+", array_content)]
#     else:
#         raise Exception(error_message + f"in file {file_path}")


"""
Input File Paths
"""
# FIXME: change
all_input_file_paths = [
    "slurm-13619553-[ST:sample_1_view_128_aug+_sd20].out",
    "slurm-13619554.out",
    "slurm-13619555.out",
    "slurm-13619556.out",
    "slurm-13619557.out",
    "slurm-13619558.out",
    "slurm-13619559.out",
    "slurm-13619560.out",
    "slurm-13619561.out",
    "slurm-13619562.out",
    "slurm-13619563.out",
    "slurm-13619564.out",
    "slurm-13619565.out",
    "slurm-13619566.out",
    "slurm-13619567.out",
    "slurm-13619568.out",
    "slurm-13619569.out",
    "slurm-13619570-[END:sample_1_view_128_aug+_sd20].out",
]

"""
Output File Paths
"""
prefix = "zz_results_"
title = "sample_1_view_128_aug+_sd20_"  # FIXME: change
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

# Channels
pattern_ideal_case_clean_channels = r"\[IDEAL CASE\]\s+\[CLEAN VAL SET\] predicted trigger channels are: tensor\(\[([\s\d,]+)\]\)"
pattern_ideal_case_poison_channels = r"\[IDEAL CASE\]\s+\[POISON VAL SET\] predicted trigger channels are: tensor\(\[([\s\d,]+)\]\)"

# Uncleansed
pattern_uncleansed_model_knn = (
    r"\[800-epoch\].*?clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
)
pattern_uncleansed_model_linear = r"for linear classifier, the ACC on clean val is: ([\d.]+), the ASR on poisoned val is: ([\d.]+)"

# Cleansed
# pattern_cleansed_model_knn = r"In kNN classification, by replacing top-\d+ channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
# pattern_cleansed_model_linear = r"In linear probe, by replacing \d+ channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"


pattern_cleansed_model_knn_remove_4 = r"In kNN classification, by replacing top-4 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_4 = r"In linear probe, by replacing 4 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_6 = r"In kNN classification, by replacing top-6 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_6 = r"In linear probe, by replacing 6 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_8 = r"In kNN classification, by replacing top-8 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_8 = r"In linear probe, by replacing 8 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_12 = r"In kNN classification, by replacing top-12 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_12 = r"In linear probe, by replacing 12 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_16 = r"In kNN classification, by replacing top-16 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_16 = r"In linear probe, by replacing 16 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_20 = r"In kNN classification, by replacing top-20 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_20 = r"In linear probe, by replacing 20 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_30 = r"In kNN classification, by replacing top-30 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_30 = r"In linear probe, by replacing 30 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_40 = r"In kNN classification, by replacing top-40 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_40 = r"In linear probe, by replacing 40 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_60 = r"In kNN classification, by replacing top-60 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_60 = r"In linear probe, by replacing 60 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_80 = r"In kNN classification, by replacing top-80 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_80 = r"In linear probe, by replacing 80 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_100 = r"In kNN classification, by replacing top-100 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_100 = r"In linear probe, by replacing 100 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"
pattern_cleansed_model_knn_remove_120 = r"In kNN classification, by replacing top-120 channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear_remove_120 = r"In linear probe, by replacing 120 channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"


output_acc_asr_file_handle = open(output_file_acc_asr, "w", encoding="utf-8")

# FIXME: update # channels removed
pattern_knn_list = [
    # pattern_cleansed_model_knn_remove_8,
    # pattern_cleansed_model_knn_remove_12,
    # pattern_cleansed_model_knn_remove_20,
    pattern_cleansed_model_knn_remove_40,
    # pattern_cleansed_model_knn_remove_60,
    # pattern_cleansed_model_knn_remove_80,
    # pattern_cleansed_model_knn_remove_100,
    # pattern_cleansed_model_knn_remove_120,
]
pattern_linear_list = [
    # pattern_cleansed_model_linear_remove_8,
    # pattern_cleansed_model_linear_remove_12,
    # pattern_cleansed_model_linear_remove_20,
    pattern_cleansed_model_linear_remove_40,
    # pattern_cleansed_model_linear_remove_60,
    # pattern_cleansed_model_linear_remove_80,
    # pattern_cleansed_model_linear_remove_100,
    # pattern_cleansed_model_linear_remove_120,
]
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

        """
        Ideal Case Voted Channels
        """
        # ideal_case_clean_channels = match_voted_channels(
        #     pattern_ideal_case_clean_channels,
        #     file_content,
        #     file_path,
        #     "no matching ideal case voted clean channels",
        # )
        # ideal_case_poison_channels = match_voted_channels(
        #     pattern_ideal_case_poison_channels,
        #     file_content,
        #     file_path,
        #     "no matching ideal case voted poison channels",
        # )
        # #  add to dictionary
        # ideal_case_channels[dataset][trigger][ssl_method][
        #     "clean"
        # ] = ideal_case_clean_channels
        # ideal_case_channels[dataset][trigger][ssl_method][
        #     "poison"
        # ] = ideal_case_poison_channels

        """
        ACC and ASR - Uncleanse
        """
        # Uncleansed kNN
        uncleansed_knn_acc, uncleansed_knn_asr = match_two_float_numbers(
            pattern_uncleansed_model_knn,
            file_content,
            file_path,
            "no matching uncleansed kNN",
        )

        # Uncleansed Linear
        uncleansed_linear_acc, uncleansed_linear_asr = match_two_float_numbers(
            pattern_uncleansed_model_linear,
            file_content,
            file_path,
            "no matching uncleansed linear",
        )

        """
        ACC and ASR - Cleansed
        """

        # FIXME: change output content
        cleansed_knn_acc_mean_list = []
        cleansed_knn_acc_std_list = []

        cleansed_knn_asr_mean_list = []
        cleansed_knn_asr_std_list = []

        cleansed_linear_acc_mean_list = []
        cleansed_linear_acc_std_list = []

        cleansed_linear_asr_mean_list = []
        cleansed_linear_asr_std_list = []

        for pattern_knn, pattern_linear in zip(pattern_knn_list, pattern_linear_list):

            # Cleansed kNN
            cleansed_knn_acc, cleansed_knn_asr = match_a_group_of_two_float_numbers(
                pattern_knn,
                file_content,
                file_path,
                "no matching cleansed kNN",
            )

            # Cleansed Linear
            cleansed_linear_acc, cleansed_linear_asr = (
                match_a_group_of_two_float_numbers(
                    pattern_linear,
                    file_content,
                    file_path,
                    "no matching cleansed linear",
                )
            )

            cleansed_knn_acc_mean_list.append(np.mean(cleansed_knn_acc))
            cleansed_knn_acc_std_list.append(np.std(cleansed_knn_acc))

            cleansed_knn_asr_mean_list.append(np.mean(cleansed_knn_asr))
            cleansed_knn_asr_std_list.append(np.std(cleansed_knn_asr))

            cleansed_linear_acc_mean_list.append(np.mean(cleansed_linear_acc))
            cleansed_linear_acc_std_list.append(np.std(cleansed_linear_acc))

            cleansed_linear_asr_mean_list.append(np.mean(cleansed_linear_asr))
            cleansed_linear_asr_std_list.append(np.std(cleansed_linear_asr))

        """
        Write Table -- collect data
        """
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"]["acc"][
            "mean"
        ] = cleansed_knn_acc_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"]["acc"][
            "std"
        ] = cleansed_knn_acc_std_list

        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"]["asr"][
            "mean"
        ] = cleansed_knn_asr_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["knn"]["asr"][
            "std"
        ] = cleansed_knn_asr_std_list

        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"]["acc"][
            "mean"
        ] = cleansed_linear_acc_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"]["acc"][
            "std"
        ] = cleansed_linear_acc_std_list

        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"]["asr"][
            "mean"
        ] = cleansed_linear_asr_mean_list
        ideal_case_acc_asr_table[dataset][trigger][ssl_method]["linear"]["asr"][
            "std"
        ] = cleansed_linear_asr_std_list

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

                        mean_val = ideal_case_acc_asr_table[dataset][trigger][method][
                            classifier
                        ][metric]["mean"][channel_count]
                        std_val = ideal_case_acc_asr_table[dataset][trigger][method][
                            classifier
                        ][metric]["std"][channel_count]

                        output_acc_asr_file_handle.write(
                            f"{mean_val:.1f}±{std_val:.1f}\t"
                        )

            output_acc_asr_file_handle.write("\n")
        output_acc_asr_file_handle.write("\n")
    output_acc_asr_file_handle.write("\n")

"""
Write Channels
"""
# # Convert to regular dict if printing or saving
# ideal_case_channels = to_dict(ideal_case_channels)

# # Output file (voted_channels)
# with open(output_file_voted_channels, "w") as f:
#     f.write("ideal_case_channels = " + repr(ideal_case_channels))
