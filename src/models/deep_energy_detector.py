from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0, InceptionV3
from tensorflow.keras.layers import Input, Conv1D, Conv2D, LSTM, Dense, Dropout, \
    MaxPooling1D, MaxPool2D, BatchNormalization, Add, Flatten


def build_seq_model(input_dim, output_dim, window_dim, custom_metrics: [],
                    loss=binary_crossentropy, learn_rate=1e-5, optimizer=None):
    """
    :param input_dim: Tensor; Dimension of input to model
    :param output_dim: Tensor; Dimension of model output
    :param window_dim: int; length of window
    :param learn_rate: float; Model learning rate
    :param custom_metrics: list; List of metrics
    :param loss: loss function; List of cost funcs
    :param optimizer: Optimizing function
    :return: tf.keras.models.Sequential; Compiled model
    """
    lr_schedule = ExponentialDecay(initial_learning_rate=learn_rate,
                                   decay_steps=10000,
                                   decay_rate=0.3)
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = optimizers.Adam(learning_rate=lr_schedule)

    model = Sequential()
    model.add(Input(shape=(window_dim, input_dim)))
    model.add(BatchNormalization())
    # model.add(Conv1D(16, (window_dim, ), padding='same', activation='relu'))
    # model.add(MaxPooling1D(padding='same'))
    model.add(LSTM(units=16, return_sequences=True, unit_forget_bias=True))
    # model.add(Conv1D(32, (window_dim,), padding='same', activation='relu'))
    # model.add(MaxPooling1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(LSTM(units=32, return_sequences=False, unit_forget_bias=False))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss=loss, metrics=custom_metrics, optimizer=optimizer)

    return model


def build_resid_model(input_dim, output_dim, window_dim, custom_metrics: [],
                      loss=binary_crossentropy, learn_rate=1e-5, optimizer=None):

    lr_schedule = ExponentialDecay(initial_learning_rate=learn_rate,
                                   decay_steps=10000,
                                   decay_rate=0.3)
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = optimizers.Adam(learning_rate=learn_rate)

    input_l = Input(shape=(window_dim, input_dim))

    # 1st block
    batch_1st = BatchNormalization()(input_l)
    conv1d_1st = Conv1D(128, (window_dim, ), padding='same', activation='relu')(batch_1st)
    conv1_mpoll_1s = MaxPooling1D(padding='same')(conv1d_1st)
    conv1d_2nd = Conv1D(64, (window_dim,), padding='same', activation='relu')(conv1_mpoll_1s)
    drop_1st = Dropout(0.5)(conv1d_2nd)
    lstm_1st = LSTM(units=3, return_sequences=False, unit_forget_bias=True)(drop_1st)

    # 2nd block
    batch_2nd = BatchNormalization()(input_l)
    lstm_2nd = LSTM(units=3, return_sequences=False, unit_forget_bias=True)(batch_2nd)

    # residual connect
    conn_node1 = Add()([lstm_1st, lstm_2nd])
    drop_1 = Dropout(0.5)(conn_node1)
    output = Dense(output_dim, activation='sigmoid')(drop_1)

    model = Model(inputs=input_l, outputs=output)
    model.compile(loss=loss, metrics=custom_metrics, optimizer=optimizer)

    return model


def build_scalogram_model(input_shape, output_dim, custom_metrics: [], default_cnn, pretrained_weights,
                          loss=binary_crossentropy, learn_rate=1e-5, optimizer=None):

    lr_schedule = ExponentialDecay(initial_learning_rate=learn_rate,
                                   decay_steps=1e6,
                                   decay_rate=3e-5)
    if optimizer is not None:
        optimizer = optimizer
    else:
        optimizer = optimizers.Adam(learning_rate=lr_schedule)

    if 'mobilenet_v2' == default_cnn:
        model = MobileNetV2(weights=pretrained_weights, input_shape=input_shape, include_top=False)
    elif 'efficientnet' == default_cnn:
        model = EfficientNetB0(weights=pretrained_weights, input_shape=input_shape, include_top=False)
    elif 'inception_v3' == default_cnn:
        model = InceptionV3(weights=pretrained_weights, input_shape=input_shape, include_top=False)
    elif 'custom' == default_cnn:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=1, kernel_size=(8, 8), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
    else:
        raise NotImplementedError

    flatten_l = Flatten()(model.output)
    drop_1 = Dropout(0.8)(flatten_l)
    output_l = Dense(output_dim, 'softmax')(drop_1)
    final_model = Model(inputs=model.inputs, outputs=output_l)

    final_model.compile(loss=loss, metrics=custom_metrics, optimizer=optimizer)

    return final_model
