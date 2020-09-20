"""
Author: Zhen Dong
Time  : 2020-09-18 10:01
"""

import numpy as np
import pandas as pd
import os
from six.moves import cPickle as pickle
from scipy import signal
import matplotlib.pyplot as plt
from mne.decoding import CSP

CHAR = ['01(B)', '02(D)', '03(G)', '04(L)', '05(O)', '06(Q)',
        '07(S)', '08(V)', '09(Z)', '10(4)', '11(7)', '12(9)']

CHAR_ROW_COL = [[1, 8], [1, 10], [2, 7], [2, 12], [3, 9], [3, 11],
                [4, 7], [4, 10], [5, 8], [5, 12], [6, 9], [6, 11]]

CHAR_ROW_COL2 = [[3, 7], [1, 12], [6, 7], [5, 10], [5, 9]]

OFFSET = 50
SCOPE = 125  # 一次刺激后的截取范围


# params:
# char - 某字符对应的全体信号数据
# event - 某字符对应的全体事件数据
# char_row_col - 对应CHAR_ROW_COL里面的某字符出现的行和列，打label用
# return:
# avg_rc - 行列对应的5次周期的平均信号数据
# labels - 标记 0或1
def five_periods_average(char, event, char_row_col):
    avg_rc = []
    labels = np.zeros((0, 1), dtype=float)
    for i in range(1, 13):
        rc = event[event[:, 0] == i]    # 每个周期1个，一共5个
        five_periods = []
        for item in rc:
            start = item[1] - 1
            tmp = char[start + OFFSET: start + SCOPE]
            five_periods.append(tmp)
        five_periods = np.array(five_periods)
        average = np.mean(five_periods, axis=0)
        temp = [average,
                1 if i in char_row_col else 0,
                i]
        avg_rc.append(temp)
        if i in char_row_col:
            labels = np.vstack((labels, [[1]] * SCOPE))
        else:
            labels = np.vstack((labels, [[0]] * SCOPE))

    return np.array(avg_rc), labels


def without_periods_average(signals, event, stimulate):
    cutout = []
    # labels = np.zeros((0, 1), dtype=float)

    for e in event:
        if 1 <= e[0] <= 12:
            start = e[1] - 1
            signal = signals[start + OFFSET: start + SCOPE]
            temp = [signal,
                    1 if e[0] in stimulate else 0,
                    e[0]]
            cutout.append(temp)

    assert len(cutout) == 60

    return np.array(cutout), []


# params:
# subject - 人的序号1～5
# return:
# all_chars - 某人的预处理后的信号数据 shape:(字符数12, 样本数150*12, 通道数20)
# labels - 标记 shape:(字符数12，样本数150*12，1)
def read_train_data(subject, is_average=True, is_save=False, part2=False):
    data_path = r'./P300/S%d/S%d_train_data.xlsx' % (subject, subject)
    event_path = r'./P300/S%d/S%d_train_event.xlsx' % (subject, subject)
    if part2:
        data_path = r'./P300/S%d/S%d_test_data.xlsx' % (subject, subject)
        event_path = r'./P300/S%d/S%d_test_event.xlsx' % (subject, subject)

    all_chars = np.zeros((0, 3), dtype=float)
    labels = []
    length = len(CHAR) if not part2 else 5
    for i in range(length):
        sheet_name = CHAR[i] if not part2 else str(13 + i)
        sheet_name = 'char' + sheet_name
        print("处理字符%s的数据..." % sheet_name)
        char_row_col = CHAR_ROW_COL[i] if not part2 else CHAR_ROW_COL2[i]
        char_i = pd.read_excel(data_path, header=None, sheet_name=sheet_name)
        event_i = pd.read_excel(event_path, header=None, sheet_name=sheet_name)
        char_i = char_i.as_matrix()
        event_i = event_i.as_matrix()

        # 巴特沃斯滤波
        char_i = butter_filter(char_i)

        # 标准化
        char_i = std(char_i, event_i)

        # 是否分段平均
        if is_average:
            res, label = five_periods_average(char_i, event_i, char_row_col)
        else:
            res, label = without_periods_average(char_i, event_i, char_row_col)

        # ICA or CSP (待完成)
        # char_i = common_spatial_pattern(char_i, event_i, char_row_col)

        all_chars = np.vstack((all_chars, res))
        labels.append(label)

    if is_save:
        save_dir = './data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        string1 = 'w' if is_average else 'wo'
        if part2:
            file_name = 's%d_train2.pkl' % subject
        else:
            file_name = 's%d_train_%s.pkl' % (subject, string1)
        save_path = os.path.join(save_dir, file_name)
        print("保存data和labels到%s..." % save_path)
        print("shape of data:", all_chars.shape)
        save_pkl({
            'data': all_chars,
            'labels': np.array(labels)
        }, save_path)

    return all_chars, np.array(labels)


def test_without_average(signals, event):
    cutout = []
    # labels = np.zeros((0, 1), dtype=float)

    for e in event:
        if 1 <= e[0] <= 12:
            start = e[1] - 1
            signal = signals[start + OFFSET: start + SCOPE]
            temp = [signal,
                    0,
                    e[0]]
            cutout.append(temp)

    assert len(cutout) == 60

    return np.array(cutout)


def std(signals, event):
    start = event[0][1] - 1
    end = event[-1][1]

    part = signals[start: end]

    for i in range(part.shape[1]):
        max = np.max(part[:, i])
        min = np.min(part[:, i])
        part[:, i] = (part[:, i] - min) / (max - min)

    signals[start: end] = part

    return signals


def read_test_data(subject, is_average=True, is_save=False):
    data_path = r'./P300/S%d/S%d_test_data.xlsx' % (subject, subject)
    event_path = r'./P300/S%d/S%d_test_event.xlsx' % (subject, subject)
    all_chars = []
    num = 10 if subject not in [2, 3] else 9

    for i in range(num):
        sheet_name = 'char' + str(i+13)
        print("处理字符%d的数据..." % (i+13))
        char_i = pd.read_excel(data_path, header=None, sheet_name=sheet_name)
        event_i = pd.read_excel(event_path, header=None, sheet_name=sheet_name)
        char_i = char_i.as_matrix()
        event_i = event_i.as_matrix()

        # 巴特沃斯滤波
        char_i = butter_filter(char_i)

        # 标准化
        char_i = std(char_i, event_i)

        # 是否分段平均
        if is_average:
            pass
            # res = test_with_average(char_i, event_i)
        else:
            res = test_without_average(char_i, event_i)

        # ICA or CSP (待完成)
        # char_i = common_spatial_pattern(char_i, event_i, char_row_col)

        all_chars.append(res)
        print(np.array(all_chars).shape)

    all_chars = np.array(all_chars)

    if is_save:
        save_dir = './data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_name = 's%d_test_%s.pkl' % (subject, 'w' if is_average else 'wo')
        save_path = os.path.join(save_dir, file_name)
        print("保存data和labels到%s..." % save_path)
        print("shape of data:", all_chars.shape)
        save_pkl({
            'data': all_chars
        }, save_path)

    return all_chars


def common_spatial_pattern(raw_data, event, pos):
    pass


def plot_data_2(signal, signal2, channel):
    x = np.linspace(0, SCOPE-1, num=SCOPE) / 250
    y1 = signal[:, channel-1]
    y2 = signal2[:, channel-1]
    minimum = np.min(y1) if np.min(y1) <= np.min(y2) else np.min(y2)
    maximum = np.max(y1) if np.max(y1) >= np.max(y2) else np.max(y2)
    plt.axis([x[0], x[len(x)-1], minimum - 10, maximum + 10])
    plt.plot(x, y1, 'g')
    plt.plot(x, y2, 'orange')
    plt.show()


def plot_data(signal, channel):
    x = np.linspace(0, SCOPE-1, num=SCOPE) / 250
    y = signal[:, channel-1]
    plt.axis([x[0], x[len(x)-1], np.min(y) - 10, np.max(y) + 10])
    plt.plot(x, y, 'g')
    plt.show()


def read_pkl(file_name):
    file_name = os.path.abspath(file_name)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data['data'], data['labels']


def save_pkl(obj, file_name):
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def butter_filter(raw_data, lowcut=25, fs=250):
    """[summary]
    Args:
        raw_data ([num, channel]): np.array,
        lowcut (int, optional): [description]. low cut frequency, don't need to revise,Defaults to 25.
        fs (int, optional): [description]. Defaults to 250. sample frequency, don't need to revise'
    Returns:
        [type]: [description]
    """
    channels = raw_data.shape[1]
    for i in range(channels):
        sos = signal.butter(N=6, Wn=lowcut, btype='lowpass', fs=fs, output='sos')
        raw_data[:, i] = signal.sosfilt(sos, raw_data[:, i])
    return raw_data


if __name__ == '__main__':
    for i in range(1, 6):
        read_train_data(subject=i, is_average=False, is_save=True, part2=False)
    # read_test_data(subject=1, is_average=False, is_save=True)

    # xx, yy = read_pkl(save_path)

    # print("被试1，行列1，通道5，字符B和L，B为绿色有P300，L为橙色无P300")
    # a1 = xx[0, 0:150, :]
    # a2 = xx[3, 0:150, :]
    # b1 = x[0, 0:150, :]
    # b2 = x[3, 0:150, :]
    # plot_data_2(a1, a2, 7)
    # plot_data_2(b1, b2, 7)
    # plot_data(a1, 7)
    # plot_data(b1, 7)
    # plot_data(a2, 7)
    # plot_data(b2, 7)
