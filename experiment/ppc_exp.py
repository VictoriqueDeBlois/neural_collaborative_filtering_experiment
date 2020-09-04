from datetime import datetime

import numpy as np

from sensitive_info import database_remote_config
from universal_method import load_training_data, csv_file_path, load_data, \
    create_sparse_matrix, auto_insert_database


def uppc_experiment(**kwargs):
    exp_data = {}
    exp_data.update(kwargs)

    sparseness = kwargs['sparseness']
    index = kwargs['index']
    extend = kwargs['extend']
    matrix_type = kwargs['matrix_type']

    matrix = create_sparse_matrix(load_training_data(sparseness, index, extend, matrix_type=matrix_type))
    test_userId, test_itemId, test_rating = \
        load_data(csv_file_path(sparseness, index, training_set=False, matrix_type=matrix_type))

    R = np.corrcoef(matrix)
    R = np.nan_to_num(R, nan=-1)

    def predict(user, item):
        i = matrix[:, item]
        i[i >= 0] = 1
        return np.matmul(R[user], matrix[:, item]) / np.sum(np.abs(np.matmul(R[user], i)))

    test_predict = np.array(list(map(lambda u, i: predict(u, i), test_userId, test_itemId)))

    mae = np.mean(np.abs(test_predict - test_rating))
    rmse = np.sqrt(np.mean(np.square(test_predict - test_rating)))

    exp_data["mae"] = float(mae)
    exp_data["rmse"] = float(rmse)
    exp_data['datetime'] = datetime.now()
    print(exp_data)
    auto_insert_database(database_remote_config, exp_data, f'uppc_{matrix_type}')


def ippc_experiment(**kwargs):
    exp_data = {}
    exp_data.update(kwargs)

    sparseness = kwargs['sparseness']
    index = kwargs['index']
    extend = kwargs['extend']
    matrix_type = kwargs['matrix_type']

    matrix = create_sparse_matrix(load_training_data(sparseness, index, extend, matrix_type=matrix_type))
    test_userId, test_itemId, test_rating = \
        load_data(csv_file_path(sparseness, index, training_set=False, matrix_type=matrix_type))

    R = np.corrcoef(matrix, rowvar=False)
    R = np.nan_to_num(R, nan=-1)

    def predict(user, item):
        u = matrix[user]
        u[i >= 0] = 1
        return np.matmul(matrix[user], R[:, item]) / np.sum(np.abs(np.matmul(u, R[:, item])))

    test_predict = np.array(list(map(lambda u, i: predict(u, i), test_userId, test_itemId)))

    mae = np.mean(np.abs(test_predict - test_rating))
    rmse = np.sqrt(np.mean(np.square(test_predict - test_rating)))

    exp_data["mae"] = float(mae)
    exp_data["rmse"] = float(rmse)
    exp_data['datetime'] = datetime.now()
    print(exp_data)
    auto_insert_database(database_remote_config, exp_data, f'ippc_{matrix_type}')


if __name__ == '__main__':
    for s in [1, 3, 5, 10, 15, 20]:
        for i in [1, 2, 3, 4, 5]:
            for e in [0]:
                exp = {
                    'sparseness': s,
                    'index': i,
                    'extend': e,
                    'matrix_type': 'tp'
                }
                uppc_experiment(**exp)
                ippc_experiment(**exp)
