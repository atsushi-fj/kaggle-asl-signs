import tensorflow as tf
from ..conv1dlstm import conv1d_lstm_block
from ...utils import load_input64_data

def get_conv1dlstm(cfg):
    X, y, _ = load_input64_data()
    
    inputs = tf.keras.layers.Input(shape=(cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    face_inputs = inputs[:, :, 0:468, :]
    left_hand_inputs = inputs[:, :, 468:489, :]
    pose_inputs = inputs[:, :, 489:522, :]
    right_hand_inputs = inputs[:, :,522:,:]
    face_vector = conv1d_lstm_block(face_inputs, [32, 64])
    left_hand_vector = conv1d_lstm_block(left_hand_inputs, [64])
    right_hand_vector = conv1d_lstm_block(right_hand_inputs, [64])
    pose_vector = conv1d_lstm_block(pose_inputs, [64])
    vector = tf.keras.layers.Concatenate(axis=1)([face_vector, left_hand_vector, right_hand_vector, pose_vector])
    vector = tf.keras.layers.Flatten()(vector)
    output = tf.keras.layers.Dense(250, activation="softmax")(vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[
            "accuracy",
        ]
    )
    return X, y, model