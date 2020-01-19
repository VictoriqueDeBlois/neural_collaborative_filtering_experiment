from keras.layers import Embedding, Input, Dense, merge, Flatten
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K


def get_model(num_users, num_items, layers=None, reg_layers=None,
              fake_layers=None, fake_reg_layers=None,
              last_activation='sigmoid', fake_last_activation='sigmoid'):
    if reg_layers is None:
        reg_layers = [0, 0]
    if layers is None:
        layers = [20, 10]
    if fake_reg_layers is None:
        fake_reg_layers = [0, 0]
    if fake_layers is None:
        fake_layers = [20, 10]
    assert len(layers) == len(reg_layers)
    assert len(fake_layers) == len(fake_reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    fake_num_layer = len(layers)

    # Input variables
    fake_user_input = Input(shape=(1,), dtype='int32', name='fake_user_input')
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    rating_output = Input(shape=(1,), dtype='float32', name='rating_output')

    MLP_Embedding_Fake_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='fake_user_embedding',
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(reg_layers[0]),
                                        input_length=1)
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # Crucial to flatten an embedding vector!
    fake_user_latent = Flatten()(MLP_Embedding_Fake_User(fake_user_input))
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    # vector = merge([user_latent, item_latent], mode = 'concat')
    vector = merge.concatenate([user_latent, item_latent])

    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation=last_activation, kernel_initializer='lecun_uniform', name='prediction')(vector)

    fake_vector = merge.concatenate([fake_user_latent, item_latent])
    for idx in range(1, fake_num_layer):
        layer = Dense(fake_layers[idx], kernel_regularizer=l2(fake_reg_layers[idx]), activation='relu',
                      name='fake_layer%d' % idx)
        fake_vector = layer(fake_vector)

    fake_prediction = Dense(1, activation=fake_last_activation, kernel_initializer='lecun_uniform',
                            name='fake_prediction')(fake_vector)

    model = Model(inputs=[fake_user_input, user_input, item_input, rating_output],
                  outputs=prediction)

    loss = K.mean(K.square(prediction - fake_prediction) + K.square(rating_output - prediction))
    model.add_loss(loss)
    model.add_metric(loss, name='loss')
    model.add_metric(K.mean(K.abs(prediction - rating_output)), name='mae')
    model.add_metric(K.sqrt(K.mean(K.square(prediction - rating_output))), name='rmse')
    return model
