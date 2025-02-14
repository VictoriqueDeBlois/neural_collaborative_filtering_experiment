from datetime import datetime

import numpy as np
from numpy import linalg as la

from sensitive_info import database_remote_config, email_config
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


def experiment(sparseness, index, percentage, extend_near_num, matrix_type):
    exp_data = {
        'sparseness': sparseness,
        'index': index,
        'extend': extend_near_num,
        'percentage': percentage
    }
    matrix = create_sparse_matrix(load_training_data(sparseness, index, extend_near_num, matrix_type=matrix_type))
    U, Sigma, VT = la.svd(matrix)
    K = sigma_pct(Sigma, percentage)
    U1 = U[:, :K]
    VT1 = VT[:K, :]
    Sigma1 = np.eye(K) * Sigma[:K]
    R = np.matmul(np.matmul(U1, Sigma1), VT1)
    mae, rmse = evaluate(sparseness, index, R, matrix_type=matrix_type)
    exp_data["mae"] = float(mae)
    exp_data["rmse"] = float(rmse)
    exp_data['datetime'] = datetime.now()
    print(exp_data)
    auto_insert_database(database_remote_config, exp_data, f'svd_{matrix_type}')
    # insert_database('experiment_data.db', "experiment_svd_rt", exp_data)


if __name__ == '__main__':
    args = []
    for s in [1, 3]:
        for i in [1]:
            for e in [0]:
                args.append((s, i, 0.1, e))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: experiment(*x), args)
        executor.shutdown(wait=True)

    send_email(receiver='haoran.x@outlook.com',
               title='SVD实验结束',
               text="",
               **email_config)
