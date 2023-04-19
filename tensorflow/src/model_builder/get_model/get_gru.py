import tensorflow as tf
import tensorflow_addons as tfa
from ...utils import load_input64_data, get_lr_metric
from ...feature import create_feature_statistics_input64
from ..gru import GRU, gru_block, mlp_block


def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, 250, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)


def get_gru(cfg):
    X, y, _ = load_input64_data()
    
    inputs = tf.keras.layers.Input(shape=(cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    x = tf.reshape(inputs, [-1, cfg.INPUT_SIZE, cfg.N_COLS * cfg.N_DIMS])
    outputs = GRU(cfg)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     clipnorm=cfg.clipnorm)
    
    lr_metric = get_lr_metric(optimizer)
    
    metrics = ["acc",lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return X, y, model


def get_feature_gru(cfg):
    X, y, _ = load_input64_data()
    feature_stats = create_feature_statistics_input64(X)
    
    inputs = tf.keras.layers.Input(shape=(cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    
    x = tf.slice(inputs, [0,0,0,0], [-1, cfg.INPUT_SIZE, cfg.N_COLS, 2])
    
    # LIPS
    lips = tf.slice(x, [0,0,feature_stats["lips"][0],0], [-1,cfg.INPUT_SIZE, 40, 2])
    lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - feature_stats["lips"][1]) / feature_stats["lips"][2],
        )
    lips = tf.reshape(lips, [-1, cfg.INPUT_SIZE, 40*2])
    
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,40,0], [-1,cfg.INPUT_SIZE, 21, 2])
    left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - feature_stats["left_hand"][1]) / feature_stats["left_hand"][2],
        )
    left_hand = tf.reshape(left_hand, [-1, cfg.INPUT_SIZE, 21*2])
    
    # POSE
    pose = tf.slice(x, [0,0,61,0], [-1,cfg.INPUT_SIZE, 5, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - feature_stats["pose"][1]) / feature_stats["pose"][2],
        )
    pose = tf.reshape(pose, [-1, cfg.INPUT_SIZE, 5*2])
    
    lips_vector = gru_block(lips, cfg)
    left_hand_vector = gru_block(left_hand, cfg)
    pose_vector = gru_block(pose, cfg)
    
    vector = tf.keras.layers.Concatenate(axis=1)([lips_vector, left_hand_vector, pose_vector])
    
    outputs = mlp_block(vector, cfg)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     clipnorm=cfg.clipnorm)
    
    lr_metric = get_lr_metric(optimizer)
    
    metrics = ["acc",lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return X, y, model


def get_new_feature_gru(cfg):
    X, y, _ = load_input64_data()
    feature_stats = create_feature_statistics_input64(X)
    
    inputs = tf.keras.layers.Input(shape=(cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    
    x = tf.slice(inputs, [0,0,0,0], [-1, cfg.INPUT_SIZE, cfg.N_COLS, 2])
    
    # LIPS
    lips = tf.slice(x, [0,0,feature_stats["lips"][0],0], [-1,cfg.INPUT_SIZE, 40, 2])
    lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - feature_stats["lips"][1]) / feature_stats["lips"][2],
        )
    lips = tf.reshape(lips, [-1, cfg.INPUT_SIZE, 40*2])
    
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,40,0], [-1,cfg.INPUT_SIZE, 21, 2])
    left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - feature_stats["left_hand"][1]) / feature_stats["left_hand"][2],
        )
    left_hand = tf.reshape(left_hand, [-1, cfg.INPUT_SIZE, 21*2])
    
    # POSE
    pose = tf.slice(x, [0,0,61,0], [-1,cfg.INPUT_SIZE, 5, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - feature_stats["pose"][1]) / feature_stats["pose"][2],
        )
    pose = tf.reshape(pose, [-1, cfg.INPUT_SIZE, 5*2])
    
    x = tf.keras.layers.Concatenate(axis=1)([lips, left_hand, pose])
    
    x = GRU(x)
    
    outputs = mlp_block(x, cfg)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     clipnorm=cfg.clipnorm)
    
    lr_metric = get_lr_metric(optimizer)
    
    metrics = ["acc",lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return X, y, model