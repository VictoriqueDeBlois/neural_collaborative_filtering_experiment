import os
from datetime import datetime

import keras
import numpy as np
from keras import metrics
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

import custom_model.GMF
from sensitive_info import database_config, email_config
from universal_method import csv_file_path, auto_insert_database, load_training_data
from universal_method import load_data, ws_num, user_num, mkdir
from universal_method import send_email


def experiment(**kwargs):
    sparseness = kwargs['sparseness']
    index = kwargs['index']
    epochs = kwargs['epochs']
    batch_size = kwargs['batch_size']
    mf_dim = kwargs['mf_dim']
    regs = kwargs['regs']
    last_activation = kwargs['last_activation']
    learning_rate = kwargs['learning_rate']
    extend_near_num = kwargs['extend_near_num']
    learner = kwargs['learner']
    matrix_type = kwargs['matrix_type']
    exp_data = {
        'sparseness': sparseness,
        'index': index,
        'epochs': epochs,
        'batch_size': batch_size,
        'mf_dim': mf_dim,
        'regs': regs,
        'last_activation': last_activation,
        'learning_rate': learning_rate,
        'extend_near_num': extend_near_num,
        'learner': learner
    }

    optimizer = {'adagrad': Adagrad(lr=learning_rate),
                 'rmsprop': RMSprop(lr=learning_rate),
                 'adam': Adam(lr=learning_rate),
                 'sgd': SGD(lr=learning_rate)}[learner]
    dataset_name = 'sparseness%s_%s' % (sparseness, index)

    model_out_file = '%s_GMF_%s_extend_%s_%s.h5' % (dataset_name, regs, extend_near_num, datetime.now())

    # load file
    userId, itemId, rating = load_training_data(sparseness, index, extend_near_num, matrix_type=matrix_type)
    test_userId, test_itemId, test_rating = \
        load_data(csv_file_path(sparseness, index, training_set=False, matrix_type=matrix_type))
    # load end

    early_stop = keras.callbacks.EarlyStopping(monitor='mean_absolute_error', min_delta=0.0002, patience=10)

    model = custom_model.GMF.get_model(num_users=user_num, num_items=ws_num, latent_dim=mf_dim, regs=regs,
                                       last_activation=last_activation)

    model.compile(optimizer=optimizer,
                  loss='mae',
                  metrics=[metrics.mae, metrics.mse])

    model.fit([userId, itemId], rating,
              batch_size=batch_size, epochs=epochs,
              callbacks=[early_stop],
              verbose=0,
              shuffle=True)

    mkdir('./Trained')
    model.save('./Trained/{}'.format(model_out_file))
    loss, mae, mse = model.evaluate([test_userId, test_itemId], test_rating, steps=1)
    # print('loss: ', loss)
    # print('mae: ', mae)
    # print('rmse', np.sqrt(mse))
    exp_data['model'] = model_out_file
    exp_data['mae'] = float(mae)
    exp_data['rmse'] = float(np.sqrt(mse))
    exp_data['datetime'] = datetime.now()
    print(exp_data)
    auto_insert_database(database_config, exp_data, f'gmf_{matrix_type}')
    return exp_data


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    all_exp_data = []
    for s in [5]:
        for i in [2, 3, 4, 5]:
            for a in ['relu', 'sigmoid']:
                for e in [0, 5]:
                    if i == 2 and a == 'relu' and e == 0:
                        continue
                    exp_data = {
                        'sparseness': s,
                        'index': i,
                        'epochs': 30,
                        'batch_size': 128,
                        'layers': [64, 32, 16],
                        'reg_layers': [0, 0, 0],
                        'last_activation': a,
                        'learning_rate': 0.007,
                        'extend_near_num': e,
                        'learner': 'adagrad'
                    }
                    all_exp_data.append(experiment(**exp_data))
    text = '\n'.join(map(lambda d: str(d), all_exp_data))
    send_email(receiver='haoran.x@outlook.com',
               title='MLP实验结束',
               text=text,
               **email_config)
