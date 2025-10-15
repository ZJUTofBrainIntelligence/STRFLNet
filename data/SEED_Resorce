##############
## SEED数据集提取每个通道5个频段的DE特征，
## 并将62个通道转化为8*9*5的三维输入，其中8*9表示62个通道转化后的二维平面，5表示5种频段
##############

import os
import sys
import math
import numpy as np
# import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from scipy.signal import butter, lfilter
from scipy.io import loadmat
from tqdm import tqdm

def decompose(file, name, split):
    # trial*channel*sample
    data = loadmat(file)
    frequency = 200

    decomposed_de = np.empty([0, split, 62])
    label = np.array([])
    all_label = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]

    for trial in range(15):
        tmp_trial_signal = data[name + '_eeg' + str(trial + 1)] #形状(62,47001)
        num_sample = int(len(tmp_trial_signal[0]) / split)
        #print('{}-{}'.format(trial + 1, num_sample))
        temp_de = np.empty([62, num_sample, split])#形状(62,470,100)
        label = np.append(label, [all_label[trial]] * num_sample)#形状(470,)

        for channel in range(62):
            trial_signal = tmp_trial_signal[channel]
            # 因为SEED数据没有基线信号部分
            win_data = np.empty([num_sample, split])
            for index in range(num_sample):
                win_data[index] = trial_signal[index * split:(index + 1) * split]#(470,100)
            temp_de[channel]=win_data#(62,470,100)
        temp_trial_de = temp_de.transpose([1, 2, 0])
        decomposed_de = np.vstack([decomposed_de, temp_trial_de])
    return decomposed_de, label

import os
import numpy as np

file_path = '../Preprocessed_EEG/'

people_name = ['1_20131027', '1_20131030', '1_20131107', '2_20140404', '2_20140413', '2_20140419',
               '3_20140603', '3_20140611', '3_20140629', '4_20140621', '4_20140702', '4_20140705',
               '5_20140411', '5_20140418', '5_20140506', '6_20130712', '6_20131016', '6_20131113',
               '7_20131027', '7_20131030', '7_20131106', '8_20140511', '8_20140514', '8_20140521',
               '9_20140620', '9_20140627', '9_20140704', '10_20131130', '10_20131204', '10_20131211',
               '11_20140618', '11_20140625', '11_20140630', '12_20131127', '12_20131201', '12_20131207',
               '13_20140527', '13_20140603', '13_20140610', '14_20140601', '14_20140615', '14_20140627',
               '15_20130709', '15_20131016', '15_20131105']

short_name = ['djc', 'djc', 'djc', 'jl', 'jl', 'jl', 'jj', 'jj', 'jj', 'lqj', 'lqj', 'lqj',
              'ly', 'ly', 'ly', 'mhw', 'mhw', 'mhw', 'phl', 'phl', 'phl', 'sxy', 'sxy', 'sxy',
              'wk', 'wk', 'wk', 'ww', 'ww', 'ww', 'wsf', 'wsf', 'wsf', 'wyw', 'wyw', 'wyw',
              'xyl', 'xyl', 'xyl', 'ys', 'ys', 'ys', 'zjy', 'zjy', 'zjy']

X = np.empty([ 0, 200, 62])
y = np.empty([ 0, 1])

for i in tqdm(range(len(people_name)), desc="Processing", ncols=100):
    file_name = file_path + people_name[i]
    #print('processing {}'.format(people_name[i]))
    decomposed_de, label = decompose(file_name, short_name[i], split=200)
    X = np.vstack([X, decomposed_de])
    y = np.append(y, label)
    #print(end="\r")
print('X:{}, Y:{}'.format(X.shape, y.shape))
print("--------------正在保存---------------")
np.save("./X_1D.npy", X)
np.save("./y.npy", y)
print("--------------保存成功---------------")
