import os
from datetime import datetime

import numpy as np
import keras
from keras import metrics
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

import custom_model.NeuMF
from experiment_structure import ExperimentData
from improve_distance_calculate import calculate_distance, convert_distance_result
from sensitive_info import database_config
from universal_method import load_csv_file, auto_insert_database
from universal_method import load_data, ws_num, user_num, mkdir


def extend_array(times: int, distance: dict, user_id: np.ndarray, item_id: np.ndarray, rating: np.ndarray):
    users = np.array(list(map(lambda x: np.append(x, np.array(distance[x])[:times][:, 0]), user_id)),
                     dtype=int).flatten()
    extend_times = list(map(lambda x: len(distance[x][:times]) + 1, user_id))
    items = np.array(list(map(lambda x, y: [x] * y, item_id, extend_times)), dtype=int).flatten()
    ratings = np.array(list(map(lambda x, y: [x] * y, rating, extend_times)), dtype=float).flatten()
    return users, items, ratings


def experiment(experiment_data: ExperimentData, last_activation):
    sparseness = experiment_data.sparseness  # 5
    index = experiment_data.data_index  # 3
    mf_dim = experiment_data.mf_dim  # 32
    epochs = experiment_data.epochs  # 30
    batch_size = experiment_data.batch_size  # 128
    layers = experiment_data.layers  # [32, 16]
    reg_layers = experiment_data.reg_layers  # [0, 0]
    learning_rate = experiment_data.learning_rate  # 0.007
    extend_near_num = experiment_data.extend_near_num  # 5
    learner = experiment_data.learner  # adagrad
    optimizer = {'adagrad': Adagrad(lr=learning_rate),
                 'rmsprop': RMSprop(lr=learning_rate),
                 'adam': Adam(lr=learning_rate),
                 'sgd': SGD(lr=learning_rate)}[learner]
    dataset_name = 'sparseness%s_%s' % (sparseness, index)
    model_out_file = '%s_NeuMF_%d_%s_%s.h5' % (dataset_name, mf_dim, layers, datetime.now())
    userId, itemId, rating = load_data(load_csv_file(sparseness, index))

    result = calculate_distance(load_csv_file(sparseness, index))
    distance = convert_distance_result(result)

    userId, itemId, rating = extend_array(extend_near_num, distance, userId, itemId, rating)

    test_userId, test_itemId, test_rating = load_data(load_csv_file(sparseness, index, training_set=False))

    early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', min_delta=0.0002, patience=10)

    model = custom_model.NeuMF.get_model(num_users=user_num, num_items=ws_num, layers=layers, reg_layers=reg_layers,
                                         mf_dim=mf_dim, last_activation=last_activation)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=[metrics.mae, metrics.mse])

    model.fit([userId, itemId], rating,
              batch_size=batch_size, epochs=epochs,
              callbacks=[early_stop],
              verbose=1,
              shuffle=True)

    mkdir('./Trained')
    model.save('./Trained/{}'.format(model_out_file))
    loss, mae, mse = model.evaluate([test_userId, test_itemId], test_rating, steps=1)
    # print('loss: ', loss)
    # print('mae: ', mae)
    # print('rmse', np.sqrt(mse))
    experiment_data.model = model_out_file
    experiment_data.loss = loss
    experiment_data.mae = mae
    experiment_data.rmse = np.sqrt(mse)
    exp_data = experiment_data.to_dict()
    exp_data['datetime'] = datetime.now()
    exp_data['last_activation'] = last_activation
    print(exp_data)
    auto_insert_database(database_config, exp_data, 'ncf_rt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for s in [5, 10]:
        for i in [1, 2, 3]:
            for a in range(2):
                for e in [0]:
                    data = ExperimentData()
                    data.sparseness = s
                    data.data_index = i
                    data.mf_dim = 8
                    data.epochs = 30
                    data.batch_size = 256
                    data.layers = [64, 32, 16, 8]
                    data.reg_layers = [0, 0, 0, 0]
                    data.learning_rate = 0.001
                    data.extend_near_num = e
                    data.learner = 'adam'
                    experiment(data, 'sigmoid' if a == 0 else 'relu')
    for s in [5]:
        for i in [2]:
            for a in range(2):
                for e in [1, 2, 3, 4, 5, 10, 15, 20]:
                    data = ExperimentData()
                    data.sparseness = s
                    data.data_index = i
                    data.mf_dim = 8
                    data.epochs = 30
                    data.batch_size = 256
                    data.layers = [64, 32, 16, 8]
                    data.reg_layers = [0, 0, 0, 0]
                    data.learning_rate = 0.001
                    data.extend_near_num = e
                    data.learner = 'adam'
                    experiment(data, 'sigmoid' if a == 0 else 'relu')


