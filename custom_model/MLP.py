"""
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
"""

from keras.layers import Embedding, Input, Dense, merge, Flatten
from keras.models import Model
from keras.regularizers import l2


def get_model(num_users, num_items, layers=None, reg_layers=None, last_activation='sigmoid'):
    if reg_layers is None:
        reg_layers = [0, 0]
    if layers is None:
        layers = [20, 10]
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # Crucial to flatten an embedding vector!
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

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model
