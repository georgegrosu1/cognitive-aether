from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, MaxPooling1D, BatchNormalization


def build_model(input_dim, output_dim, window_dim, custom_metrics: [],
                loss=binary_crossentropy, optimizer=None):
    """
    :param input_dim: Tensor; Dimension of input to model
    :param output_dim: Tensor; Dimension of model output
    :param window_dim: int; length of window
    :param custom_metrics: list; List of metrics
    :param loss: loss function; List of cost funcs
    :param optimizer: Optimizing function
    :return: tf.keras.models.Sequential; Compiled model
    """
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-5,
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
