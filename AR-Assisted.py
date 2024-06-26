import pandas as pd
from pandas import read_csv
from itertools import product
import random

# define seq_modes
# 'I': Increase
# 'D': Decrease
# 'L': Level
# 'N': Nan-Nan

# 'O': Others

def seq2mode(seq_df, col):
    df_convert_ = seq_df.copy()
    for i in range(len(df_convert_) - 1):
        if df_convert_.iloc[i, col] == 0:
            if df_convert_.iloc[i + 1, 0] == 0:
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


def mode2string(mode_df, col):
    mode_df.iloc[:, col] = mode_df.iloc[:, col].astype(str)
    return mode_df.iloc[:-1, col].str.cat(sep='')

def find_all_occurrences(s, sub):
    occurrences = []
    start = 0
    while True:
        start = s.find(sub, start)
        if start == -1:
            break
        occurrences.append(start)
        start += len(sub)
    return occurrences


def generate_permutations(characters, length):
    permutations = [''.join(p) for p in product(characters, repeat=length)]
    return permutations


def find_frequencies(str_input, size, permutations):
    ret = []
    for s in permutations:
        positions = find_all_occurrences(str_input, s)
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
        print('No frequencies matched')
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


def get_label_outer():
    rand = random.randint(1, 2)
    if rand == 1:
        return 'I'
    else:
        return 'D'

dataset = read_csv(
    'all_data_weekly.csv',
    header=0)

df_transposed = dataset.transpose()

df_transposed.reset_index(inplace=True)

new_columns = df_transposed.iloc[0]
df_transposed = df_transposed[1:]
df_transposed.columns = new_columns
# df_transposed = df_transposed.set_index('date')
print(df_transposed)

df_train = df_transposed.loc[0:104]
pred_len = 12
df_real = df_transposed.loc[105:105 + pred_len - 1].reset_index(drop=True)

print(df_train)
print(df_real)

characters = ['N', 'I', 'D', 'L']
permutations_w = generate_permutations(characters, 5)

pred_ratio_all = []
pred_label_all = []

freq_th_list = [2, 3, 4, 5]
for freq_th in freq_th_list:
    pred_ratio_all = []
    pred_label_all = []
    for col_i in range(df_train.shape[1] - 1):
        weekly_df_convert = seq2mode(df_train, col_i + 1)
        big_string_weekly = mode2string(weekly_df_convert, col_i + 1)
        result_w = find_frequencies(big_string_weekly, freq_th, permutations_w)
        # print(result_w)
        end_string = big_string_weekly[-4:]

        pred_ratio = []
        pred_ratio.append(weekly_df_convert.columns[col_i + 1])
        pred_label = []
        for pred_i in range(pred_len):
            if find_matched_frequencies_to_seq(end_string, result_w, df_train, col_i + 1):
                frequent_item_set, freq_seq = find_matched_frequencies_to_seq(end_string, result_w, df_train, col_i + 1)
                ratio_list, ratio_ret, label_ret = clac_pred_ratio(freq_seq)
                pred_label.append(label_ret)
                pred_ratio.append(ratio_ret)

                end_string = end_string[1:] + label_ret
            else:
                # print(f'{col_i}-{pred_i}-Notice: get_label_outer')
                end_string = end_string[1:] + get_label_outer()

        # plt.plot(pred_ratio)
        pred_ratio_all.append(pred_ratio)

    pred_ratio_AR_df = pd.DataFrame(pred_ratio_all, columns=None)
    pred_ratio_AR_df.to_csv(f'pred_ratio_AR-{pred_len}-{freq_th}.csv', index=False)
