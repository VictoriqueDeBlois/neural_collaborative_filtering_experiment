from keras.layers import Embedding, Input, Dense, merge, Flatten
from keras.models import Model
from keras.regularizers import l2


def get_model(num_users, num_items, latent_dim, regs=None, last_activation='sigmoid'):
    # Input variables
    if regs is None:
        regs = [0, 0]
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding',
    # init = init_normal, W_regularizer = l2(regs[0]), input_length=1)
    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(regs[0]),
                                  input_length=1)
    # MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding',
    # init = init_normal, W_regularizer = l2(regs[1]), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(regs[1]),
                                  input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings
    # predict_vector = merge([user_latent, item_latent], mode = 'mul')
    predict_vector = merge.multiply([user_latent, item_latent])
    # predict_vector = merge.dot([user_latent, item_latent], 1)
    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation=last_activation, kernel_initializer='lecun_uniform', name='prediction')(predict_vector)

    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)

    return model
