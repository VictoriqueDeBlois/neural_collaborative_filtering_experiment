import sqlite3
from datetime import datetime

import numpy as np
from sklearn.decomposition import NMF

from sensitive_info import database_config
from universal_method import load_csv_file, auto_insert_database, evaluate, create_sparse_matrix


def experiment(sparseness, data_index, K, steps=5000, alpha=0.0002):
    exp_data = {
        'sparseness': sparseness,
        'data_index': data_index,
        'K': K,
        'steps': steps,
        'alpha': alpha
    }
    R = create_sparse_matrix(load_csv_file(sparseness, data_index))
    model = NMF(n_components=K, alpha=alpha, max_iter=steps)
    w = model.fit_transform(R)
    h = model.components_
    X = np.matmul(w, h)
    mae, rmse = evaluate(sparseness, data_index, X)
    exp_data["mae"] = float(mae)
    exp_data["rmse"] = float(rmse)
    exp_data['datetime'] = datetime.now()
    print(exp_data)
    auto_insert_database(database_config, exp_data, 'mf_rt')
    # insert_database('experiment_data.db', 'experiment_mf_rt', exp_data)


def create_experiment_database(path='experiment_data.db'):
    conn = sqlite3.connect(path)
    try:
        conn.execute('''create table experiment_mf_rt(
                        id int identity(1, 1) primary key,
                        sparseness int,
                        data_index int,
                        K int,
                        steps int,
                        alpha double,
                        mae double,
                        rmse double
                        )''')
    except sqlite3.Error as error:
        print(error)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    for k in range(2, 10):
        for s in [5, 10, 15, 20, 30]:
            for i in range(1, 6):
                experiment(s, i, k)
