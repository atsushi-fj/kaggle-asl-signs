import tensorflow as tf


def conv1d_lstm_block(inputs, filters):
    vector = tf.keras.layers.ConvLSTM1D(filters=32, kernel_size=8)(inputs)
    for f in filters:
        vector = tf.keras.layers.Conv1D(filters=f, kernel_size=8)(vector)
        vector = tf.keras.layers.MaxPooling1D()(vector)
    vector = tf.keras.layers.Dropout(0.3)(vector)
    return vector

