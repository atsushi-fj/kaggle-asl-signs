import tensorflow as tf
from ..conv1dlstm import conv1d_lstm_block
from ...utils import load_input64_data
from ...feature import create_feature_statistics_input64

def get_conv1dlstm(cfg):
    X, y, _ = load_input64_data()
    feature_stats = create_feature_statistics_input64(X)
    
    inputs = tf.keras.layers.Input(shape=(cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    x = inputs
    
    # LIPS
    lips = tf.slice(x, [0,0,feature_stats["lips"][0],0], [-1,cfg.INPUT_SIZE, 40, 2])
    lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - feature_stats["lips"][1]) / feature_stats["lips"][2],
        )
    lips = tf.reshape(lips, [-1, cfg.INPUT_SIZE, 40, 2])
    
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,40,0], [-1,cfg.INPUT_SIZE, 21, 2])
    left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - feature_stats["left_hand"][1]) / feature_stats["left_hand"][2],
        )
    left_hand = tf.reshape(left_hand, [-1, cfg.INPUT_SIZE, 21, 2])
    
    # POSE
    pose = tf.slice(x, [0,0,61,0], [-1,cfg.INPUT_SIZE, 5, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - feature_stats["pose"][1]) / feature_stats["pose"][2],
        )
    pose = tf.reshape(pose, [-1, cfg.INPUT_SIZE, 5, 2])
    
    face_vector = conv1d_lstm_block(lips, [64])
    left_hand_vector = conv1d_lstm_block(left_hand, [64])
    pose_vector = conv1d_lstm_block(pose, [64])
    vector = tf.keras.layers.Concatenate(axis=1)([face_vector, left_hand_vector, pose_vector])
    vector = tf.keras.layers.Flatten()(vector)
    output = tf.keras.layers.Dense(250, activation="softmax")(vector)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[
            "accuracy",
        ]
    )
    return X, y, model