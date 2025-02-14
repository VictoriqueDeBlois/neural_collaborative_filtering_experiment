from datetime import datetime

import numpy as np
from sklearn.decomposition import NMF

from sensitive_info import database_config
from universal_method import auto_insert_database, evaluate, create_sparse_matrix, load_training_data


def experiment(sparseness, data_index, K, steps=5000, alpha=0.0002):
    exp_data = {
        'sparseness': sparseness,
        'data_index': data_index,
        'K': K,
        'steps': steps,
        'alpha': alpha
    }
    extend_near_num = 0
    R = create_sparse_matrix(load_training_data(sparseness, data_index, extend_near_num))
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


if __name__ == '__main__':
    for k in range(2, 10):
        for s in [5, 10, 15, 20, 30]:
            for i in range(1, 6):
                experiment(s, i, k)
