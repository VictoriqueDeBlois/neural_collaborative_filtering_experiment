from datetime import datetime

import numpy as np
from numpy import linalg as la

from sensitive_info import database_config, email_config
from universal_method import auto_insert_database, \
    create_sparse_matrix, evaluate, load_training_data, send_email


def sigma_pct(sigma, percentage):
    sigma_square = np.square(sigma)
    threshold = np.sum(sigma_square) * percentage
    sigma_square_sum = 0
    k = 0
    for i in sigma_square:
        sigma_square_sum += i
        k += 1
        if sigma_square_sum >= threshold:
            return k


def experiment(sparseness, index, percentage, extend_near_num):
    exp_data = {
        'sparseness': sparseness,
        'data_index': index,
        'extend_near_num': extend_near_num,
        'percentage': percentage
    }
    matrix = create_sparse_matrix(load_training_data(sparseness, index, extend_near_num))
    U, Sigma, VT = la.svd(matrix)
    K = sigma_pct(Sigma, percentage)
    U1 = U[:, :K]
    VT1 = VT[:K, :]
    Sigma1 = np.eye(K) * Sigma[:K]
    R = np.matmul(np.matmul(U1, Sigma1), VT1)
    mae, rmse = evaluate(sparseness, index, R)
    exp_data["mae"] = float(mae)
    exp_data["rmse"] = float(rmse)
    exp_data['datetime'] = datetime.now()
    # print(exp_data)
    auto_insert_database(database_config, exp_data, 'svd_rt')
    # insert_database('experiment_data.db', "experiment_svd_rt", exp_data)


if __name__ == '__main__':
    for s in [5, 10, 15, 20]:
        for i in range(1, 6):
            for e in [0, 1, 2, 3, 4, 5, 10, 15, 20]:
                experiment(s, i, 0.1, e)
    send_email(receiver='haoran.x@outlook.com',
               title='SVD实验结束',
               text="",
               **email_config)
