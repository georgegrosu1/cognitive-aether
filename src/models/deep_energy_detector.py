from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, MaxPooling1D


def build_model(input_dim, output_dim, num_features, metrics: list, loss='sigmoid'):
    lr_schedule = ExponentialDecay(initial_learning_rate=1e-5,
                                   decay_steps=10000,
                                   decay_rate=0.7)
    optimizer = optimizers.Adam(learning_rate=lr_schedule)

    model = Sequential()
    model.add(Input(shape=(input_dim, num_features)))
    model.add(LSTM(units=64, return_state=True, unit_forget_bias=True))
    model.add(LSTM(units=64, return_state=False, unit_forget_bias=True))
    model.add(Dense(32, activation='swish'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
