import keras
from keras.initializers import glorot_uniform
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2


def simple_ncf(num_user, num_item, mf_dim, layers, reg_layers):
    # 潜在特征
    hid_f = mf_dim
    # 网络单元数
    units = layers
    reg_p = reg_layers[0]
    # drop_p = self.creParm.drop_p

    input_u = Input(shape=(1,), dtype="int32")
    input_s = Input(shape=(1,), dtype="int32")
    # print(input_u)
    # one hot 转换
    # u_hid = HidFeatLayer(num_user, hid_f)(input_u)
    # s_hid = HidFeatLayer(num_item, hid_f)(input_s)
    u_hid = Embedding(input_dim=num_user, output_dim=hid_f)(input_u)
    s_hid = Embedding(input_dim=num_item, output_dim=hid_f)(input_s)
    u_hid = Flatten()(u_hid)
    s_hid = Flatten()(s_hid)
    # print(u_hid, s_hid)
    # 连接
    out = Concatenate()([u_hid, s_hid])
    # print(out)

    # 若干全连接层和dropout
    for unit in units:
        out = Dense(unit,
                    activation=keras.activations.relu,
                    kernel_initializer=glorot_uniform(),
                    kernel_regularizer=l2(reg_p))(out)
        # out = Dropout(drop_p)(out)

    # print(out)

    out = Dense(1,
                activation=keras.activations.relu,
                kernel_initializer=glorot_uniform(),
                kernel_regularizer=l2(reg_p))(out)
    # print(out)
    return Model(inputs=[input_u, input_s], outputs=out)
