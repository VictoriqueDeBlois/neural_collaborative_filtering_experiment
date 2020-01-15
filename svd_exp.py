import sqlite3
from datetime import datetime

import numpy as np
from numpy import linalg as la

from sensitive_info import database_config
from universal_method import load_csv_file, auto_insert_database, \
    create_sparse_matrix, evaluate


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


def experiment(sparseness, index, percentage):
    exp_data = {
        'sparseness': sparseness,
        'data_index': index,
        'percentage': percentage
    }
    matrix = create_sparse_matrix(load_csv_file(sparseness, index))
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
    print(exp_data)
    auto_insert_database(database_config, exp_data, 'svd_rt')
    # insert_database('experiment_data.db', "experiment_svd_rt", exp_data)


def create_experiment_database(path='experiment_data.db'):
    conn = sqlite3.connect(path)
    try:
        conn.execute('''create table experiment_svd_rt(
                        id int identity(1, 1) primary key,
                        sparseness int,
                        data_index int,
                        percentage double,
                        mae double,
                        rmse double
                        )''')
    except sqlite3.Error as error:
        print(error)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    for s in [5, 10, 15, 20]:
        for i in range(1, 6):
            experiment(s, i, 0.1)
