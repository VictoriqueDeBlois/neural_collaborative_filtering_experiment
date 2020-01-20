import os
from datetime import datetime

import numpy as np
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

import custom_model.extend_mlp_model
from improve_distance_calculate import calculate_distance, convert_distance_result
from sensitive_info import database_config
from universal_method import load_csv_file, auto_insert_database
from universal_method import load_data, ws_num, user_num, mkdir


def extend_array(times: int, distance: dict, user_id: np.ndarray, item_id: np.ndarray, rating: np.ndarray):
    fake_users = np.array(list(map(lambda x: np.array(distance[x])[:times][:, 0], user_id)), dtype=int).flatten()
    extend_times = list(map(lambda x: len(distance[x][:times]), user_id))
    users = np.array(list(map(lambda x, y: [x] * y, user_id, extend_times)), dtype=int).flatten()
    items = np.array(list(map(lambda x, y: [x] * y, item_id, extend_times)), dtype=int).flatten()
    ratings = np.array(list(map(lambda x, y: [x] * y, rating, extend_times)), dtype=float).flatten()
    return fake_users, users, items, ratings


def experiment(**kwargs):
    sparseness = kwargs['sparseness']
    index = kwargs['index']
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    layers = kwargs['layers']
    reg_layers = kwargs['reg_layers']
    fake_layers = kwargs['fake_layers']
    fake_reg_layers = kwargs['fake_reg_layers']
    last_activation = kwargs['last_activation']
    fake_last_activation = kwargs['fake_last_activation']
    learning_rate = kwargs['learning_rate']
    extend_near_num = kwargs['extend_near_num']
    learner = kwargs['learner']
    exp_data = {
        'sparseness': sparseness,
        'index': index,
        'epochs': epochs,
        'batch_size': batch_size,
        'layers': layers,
        'reg_layers': reg_layers,
        'fake_layers': fake_layers,
        'fake_reg_layers': fake_reg_layers,
        'last_activation': last_activation,
        'fake_last_activation': fake_last_activation,
        'learning_rate': learning_rate,
        'extend_near_num': extend_near_num,
        'learner': learner
    }

    optimizer = {'adagrad': Adagrad(lr=learning_rate),
                 'rmsprop': RMSprop(lr=learning_rate),
                 'adam': Adam(lr=learning_rate),
                 'sgd': SGD(lr=learning_rate)}[learner]
    dataset_name = 'sparseness%s_%s' % (sparseness, index)
    model_out_file = '%s_exMLP_%s_%s_%s.h5' % (dataset_name, layers, fake_layers, datetime.now())

    userId, itemId, rating = load_data(load_csv_file(sparseness, index))

    result = calculate_distance(load_csv_file(sparseness, index))
    distance = convert_distance_result(result)

    fake_user_id, userId, itemId, rating = extend_array(extend_near_num, distance, userId, itemId, rating)

    test_userId, test_itemId, test_rating = load_data(load_csv_file(sparseness, index, training_set=False))

    # early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', min_delta=0.0002, patience=10)

    model = custom_model.extend_mlp_model.get_model(num_users=user_num, num_items=ws_num,
                                                    layers=layers, reg_layers=reg_layers,
                                                    fake_layers=fake_layers, fake_reg_layers=fake_reg_layers,
                                                    last_activation=last_activation,
                                                    fake_last_activation=fake_last_activation)

    model.compile(optimizer=optimizer)

    model.fit(x=[fake_user_id, userId, itemId, rating],
              y=None,
              batch_size=batch_size, epochs=epochs,
              verbose=1,
              shuffle=False)
    mkdir('./Trained')
    model.save('./Trained/{}'.format(model_out_file))
    # prediction, fake_prediction = model.predict([np.zeros(len(test_userId)), test_userId, test_itemId])
    # mae = np.mean(np.abs(prediction - test_rating))
    # rmse = np.sqrt(np.mean(np.square(prediction - test_rating)))

    _, _, mae, rmse = model.evaluate([np.zeros(len(test_rating)), test_userId, test_itemId, test_rating], steps=1)
    # print('loss: ', loss)
    # print('mae: ', mae)
    # print('rmse', np.sqrt(mse))
    exp_data['model'] = model_out_file
    exp_data['mae'] = float(mae)
    exp_data['rmse'] = float(rmse)
    exp_data['datetime'] = datetime.now()
    exp_data['last_activation'] = last_activation
    print(exp_data)
    auto_insert_database(database_config, exp_data, 'exmlp_rt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for s in [5]:
        for i in [2]:
            for a in ['relu', 'sigmoid']:
                for e in [1, 2, 3, 4, 5, 10, 15]:
                    exp_data = {
                        'sparseness': s,
                        'index': i,
                        'epochs': 50,
                        'batch_size': 256,
                        'layers': [64, 32, 16, 8],
                        'reg_layers': [0, 0, 0, 0],
                        'fake_layers': [64, 32, 16, 8],
                        'fake_reg_layers': [0, 0, 0, 0],
                        'last_activation': a,
                        'fake_last_activation': a,
                        'learning_rate': 0.001,
                        'extend_near_num': e,
                        'learner': 'adam'
                    }
                    experiment(**exp_data)
