# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:34:38 2016

@author:
"""
import os
import random

import numpy as np
from universal_method import mkdir, load_original_matrix_data


def no_coincide_sample(data, pairs, ration, train_file, test_file):
    sample_num = len(pairs) * ration
    sample_pairs = random.sample(pairs, int(sample_num))
    train_pairs = random.sample(sample_pairs, int(sample_num * 0.9))
    test_pairs = np.array(list(set(sample_pairs) - set(train_pairs)))
    train_pairs = np.array(train_pairs)
    train_data = np.array([train_pairs[:, 0], train_pairs[:, 1], data[train_pairs[:, 0], train_pairs[:, 1]]]).T
    test_data = np.array([test_pairs[:, 0], test_pairs[:, 1], data[test_pairs[:, 0], test_pairs[:, 1]]]).T
    np.savetxt(train_file, train_data, fmt='%s', delimiter='\t')
    np.savetxt(test_file, test_data, fmt='%s', delimiter='\t')


def screen_pairs(data):
    pairs = np.argwhere(data >= 0)
    pairs_set = set()
    for pair in pairs:
        pairs_set.add(tuple(pair))
    return pairs_set


def create_sample(matrix, sparseness, division_num=5):
    save_root_path = './CSV_{}'.format(matrix)
    mkdir(save_root_path)
    matrix_data = load_original_matrix_data('./Data/{}Matrix.txt'.format(matrix))
    pairs = screen_pairs(matrix_data)
    for ration in sparseness:
        save_path = os.path.join(save_root_path, 'sparseness%d' % ration)
        mkdir(save_path)
        for fileNum in range(1, division_num + 1):
            print(ration, fileNum)
            file_name_training = 'training{}.csv'.format(fileNum)
            file_name_training = os.path.join(save_path, file_name_training)
            file_name_test = 'test{}.csv'.format(fileNum)
            file_name_test = os.path.join(save_path, file_name_test)
            no_coincide_sample(matrix_data, pairs, ration / 100.0, file_name_training, file_name_test)


if __name__ == '__main__':
    create_sample('rt', [5, 10, 15, 20])
