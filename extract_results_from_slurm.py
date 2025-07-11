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
def match_two_float_numbers(pattern, file_content, file_path, error_message):
    match_pattern = re.search(pattern, file_content)
    if match_pattern:
        result_1 = float(match_pattern.group(1))
        result_2 = float(match_pattern.group(2))
        return result_1, result_2
    else:
        raise Exception(error_message + f"in file {file_path}")


# match basic information such as trigger type
def match_basic_info(pattern, file_content, file_path, error_message):
    match_pattern = re.search(pattern, file_content)
    if match_pattern:
        return match_pattern.group(1)
    else:
        raise Exception(error_message + f"in file {file_path}")


# match voted channels
def match_voted_channels(pattern, file_content, file_path, error_message):
    match_pattern = re.search(pattern, file_content)
    if match_pattern:
        array_content = match_pattern.group(1)
        return [int(num) for num in re.findall(r"\d+", array_content)]
    else:
        raise Exception(error_message + f"in file {file_path}")


"""
Input File Paths
"""
# FIXME: change
all_input_file_paths = [
    "slurm-12685211-[ST:sd20_ideal].out",
    "slurm-12685212.out",
    "slurm-12685213.out",
    "slurm-12685214.out",
    "slurm-12685215.out",
    "slurm-12685216.out",
    "slurm-12685217.out",
    "slurm-12685218.out",
    "slurm-12685219.out",
    "slurm-12685220.out",
    "slurm-12685221.out",
    "slurm-12685222.out",
    "slurm-12685223.out",
    "slurm-12685224.out",
    "slurm-12685225.out",
    "slurm-12685226.out",
    "slurm-12685227.out",
    "slurm-12685228-[END:sd20_ideal].out",
]

"""
Output File Paths
"""
prefix = "zz_results_"
title = "sd20_ideal_case_"  # FIXME: change
output_file_voted_channels = prefix + title + "voted_channels.py"
output_file_acc_asr = prefix + title + "acc_asr_results.txt"

ideal_case_channels = nested_dict()


"""
Regex Patterns
"""
pattern_dataset = r"dataset: (.*)"
pattern_trigger = r"trigger_type: (.*)"
pattern_ssl_method = r"method: (.*)"
# [\d.]+ matches a number that may include a decimal point.
pattern_ideal_case_clean_channels = r"\[IDEAL CASE\]\s+\[CLEAN VAL SET\] predicted trigger channels are: tensor\(\[([\s\d,]+)\]\)"
pattern_ideal_case_poison_channels = r"\[IDEAL CASE\]\s+\[POISON VAL SET\] predicted trigger channels are: tensor\(\[([\s\d,]+)\]\)"
pattern_uncleansed_model_knn = (
    r"\[800-epoch\].*?clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
)
pattern_uncleansed_model_linear = r"for linear classifier, the ACC on clean val is: ([\d.]+), the ASR on poisoned val is: ([\d.]+)"
pattern_cleansed_model_knn = r"In kNN classification, by replacing top-\d+ channels, clean acc:\s*([\d.]+)\s*\|\s*back acc:\s*([\d.]+)"
pattern_cleansed_model_linear = r"In linear probe, by replacing \d+ channels, the ACC on clean val is:\s*([\d.]+), the ASR on poisoned val is:\s*([\d.]+)"

output_acc_asr_file_handle = open(output_file_acc_asr, "w")

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
        ideal_case_clean_channels = match_voted_channels(
            pattern_ideal_case_clean_channels,
            file_content,
            file_path,
            "no matching ideal case voted clean channels",
        )
        ideal_case_poison_channels = match_voted_channels(
            pattern_ideal_case_poison_channels,
            file_content,
            file_path,
            "no matching ideal case voted poison channels",
        )
        #  add to dictionary
        ideal_case_channels[dataset][trigger][ssl_method][
            "clean"
        ] = ideal_case_clean_channels
        ideal_case_channels[dataset][trigger][ssl_method][
            "poison"
        ] = ideal_case_poison_channels

        """
        ACC and ASR
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

        # Cleansed kNN
        cleansed_knn_acc, cleansed_knn_asr = match_two_float_numbers(
            pattern_cleansed_model_knn,
            file_content,
            file_path,
            "no matching cleansed kNN",
        )

        # Cleansed Linear
        cleansed_linear_acc, cleansed_linear_asr = match_two_float_numbers(
            pattern_cleansed_model_linear,
            file_content,
            file_path,
            "no matching cleansed linear",
        )

        """
        Write ACC and ASR results to txt file
        """
        output_acc_asr_file_handle.write(
            f"------------\n{dataset:<6}\t{trigger:<6}\t{ssl_method}\n------------\n"
        )
        output_acc_asr_file_handle.write(
            f"{'Uncleansed:':<15} kNN: {uncleansed_knn_acc:<5} {uncleansed_knn_asr:<5} Linear: {uncleansed_linear_acc:<5} {uncleansed_linear_asr:<5}\n"
        )
        output_acc_asr_file_handle.write(
            f"{'Cleansed:':<15} kNN: {cleansed_knn_acc:<5} {cleansed_knn_asr:<5} Linear: {cleansed_linear_acc:<5} {cleansed_linear_asr:<5}\n"
        )

# Convert to regular dict if printing or saving
ideal_case_channels = to_dict(ideal_case_channels)

# Output file (voted_channels)
with open(output_file_voted_channels, "w") as f:
    f.write("ideal_case_channels = " + repr(ideal_case_channels))
