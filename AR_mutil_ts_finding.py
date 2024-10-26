import random
from itertools import product
from tqdm import tqdm

import numpy as np
import pandas as pd
from pandas import read_csv

# define seq_modes
# 'I': Increase
# 'D': Decrease
# 'L': Level
# 'N': Nan-Nan

# 'O': Others
def seq2mode_cluster(seq_df):
    df_convert_ = seq_df.copy()
    for col in range(1, seq_df.shape[1]):
        for i in range(len(df_convert_) - 1):
            if df_convert_.iloc[i, col] == 0:
                if df_convert_.iloc[i + 1, col] == 0:
                    df_convert_.iloc[i, col] = 'N'
                else:
                    df_convert_.iloc[i, col] = 'I'
            elif df_convert_.iloc[i, col] == df_convert_.iloc[i + 1, col]:
                df_convert_.iloc[i, col] = 'L'
            elif df_convert_.iloc[i, col] > df_convert_.iloc[i + 1, col]:
                df_convert_.iloc[i, col] = 'D'
            elif df_convert_.iloc[i, col] < df_convert_.iloc[i + 1, col]:
                df_convert_.iloc[i, col] = 'I'
            else:
                seq_df.iloc[i, col] = 'O'
    return df_convert_


def mode2string_cluster(mode_df):
    str_list = []
    for col in range(1, mode_df.shape[1]):
        mode_df.iloc[:, col] = mode_df.iloc[:, col].astype(str)
        str_list.append(mode_df.iloc[:-1, col].str.cat(sep=''))
    return str_list


def find_all_occurrences_cluster(s_list, sub):
    occurrences = []
    for i in range(1, len(s_list)):
        s = s_list[i]
        start = 0
        while True:
            start = s.find(sub, start)
            if start == -1:
                break
            occurrences.append(start)
            start += len(sub)  # Move start index past this occurrence
    return occurrences


def generate_permutations(characters, length):
    permutations = [''.join(p) for p in product(characters, repeat=length)]
    return permutations


def find_frequencies_cluster(str_list_input, size, permutations):
    ret = []
    for s in permutations:
        positions = find_all_occurrences_cluster(str_list_input, s)
        if len(positions) > size:
            ret.append([s, len(positions), positions])
            # print(s, len(positions), positions)
    return ret


def find_matched_frequencies_to_seq(freq_mode, freq_item_set, df_ori, col):
    frequent_item_set = []
    final_str_mode_test = freq_mode
    for r in freq_item_set:
        if r[0][0:len(freq_mode)] == final_str_mode_test:
            frequent_item_set.append(r)

    if len(frequent_item_set) == 0:
        # print('No frequencies matched')
        return False

    freq_seq = []
    for f in frequent_item_set:
        mode_size = len(f[0])
        append_temp = []
        for i in range(f[1]):
            append_temp.append(df_ori.iloc[f[2][i]:f[2][i] + mode_size + 1, col].values.tolist())
        freq_seq.append(append_temp)

    return frequent_item_set, freq_seq


def calculate_average(lst):
    non_zero_elements = [x for x in lst if x != 0]
    if len(non_zero_elements) == 0:
        return 0

    average = sum(non_zero_elements) / len(non_zero_elements)
    return average


def clac_pred_ratio(freq_seq):
    label = 'N'
    if len(freq_seq) == 0:
        return 0

    ratio_list = []
    for i in range(len(freq_seq)):
        ratio_temp = []
        for j in range(len(freq_seq[i])):
            if freq_seq[i][j][-2] >= freq_seq[i][j][-1] and freq_seq[i][j][-1] != 0:
                ratio_temp.append((freq_seq[i][j][-1] - freq_seq[i][j][-2]) / freq_seq[i][j][-2])
            elif freq_seq[i][j][-2] < freq_seq[i][j][-1] and freq_seq[i][j][-2] != 0:
                ratio_temp.append(((freq_seq[i][j][-1]) - (freq_seq[i][j][-2])) / freq_seq[i][j][-1])
            else:
                ratio_temp.append(0)

        ratio_list.append(ratio_temp)

    freq_mode_len_list = []
    for i in range(len(freq_seq)):
        freq_mode_len_list.append(len(freq_seq[i]))

    ratio_ret = 0
    for i in range(len(ratio_list)):
        ratio_mean = calculate_average(ratio_list[i])
        ratio_ret += ratio_mean * (freq_mode_len_list[i] / sum(freq_mode_len_list))

    if ratio_ret > 0:
        label = 'I'
    elif ratio_ret < 0:
        label = 'D'
    else:
        label = 'L'

    return ratio_list, ratio_ret, label



def find_result_w(result_w_list, cur_PN, cluster_detail):
    result_w_i = cluster_detail[cluster_detail['PN'] == cur_PN].iloc[0,1]
    return result_w_list[result_w_i - 1]
    pass


def filter_top_by_threshold(data, threshold):
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

    num_items = int(len(sorted_data) * threshold)

    return sorted_data[:num_items]
