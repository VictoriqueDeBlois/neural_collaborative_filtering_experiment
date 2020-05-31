from keras.layers import Embedding, Input, Dense, merge, Flatten
from keras.models import Model
from keras.regularizers import l2


def get_model(num_users, num_items, mf_dim=10, layers=None, reg_layers=None, reg_mf=0, last_activation='sigmoid'):
    if reg_layers is None:
        reg_layers = [0]
    if layers is None:
        layers = [10]
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layer
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf),
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_mf),
                                  input_length=1)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name="mlp_embedding_user",
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='mlp_embedding_item',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg_layers[0]),
                                   input_length=1)

    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    # mf_vector = merge([mf_user_latent, mf_item_latent], mode='mul')  # element-wise multiply
    # mf_vector = merge.dot([mf_user_latent, mf_item_latent], 1)
    mf_vector = merge.multiply([mf_user_latent, mf_item_latent])

    # MLP part
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    # mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode='concat')
    mlp_vector = merge.concatenate([mlp_user_latent, mlp_item_latent])

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name="layer%d" % idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    # predict_vector = merge([mf_vector, mlp_vector], mode='concat')
    predict_vector = merge.concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation=last_activation, kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model
