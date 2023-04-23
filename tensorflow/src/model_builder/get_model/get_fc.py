import tensorflow as tf
import tensorflow_addons as tfa
from ...utils import load_ln_data, get_lr_metric


def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, 250, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)


def get_fc(cfg):
    X, y = load_ln_data()
    
    inputs = tf.keras.layers.Input(shape=(cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    x = tf.reshape(inputs, [-1, cfg.N_COLS*cfg.N_DIMS])
    
    fc_layer = tf.keras.Sequential([
        tf.keras.layers.Dense(cfg.UNITS),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED),
        tf.keras.layers.Dense(cfg.UNITS),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED),
        tf.keras.layers.Dense(cfg.UNITS),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED),
        tf.keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax")
    ])
    outputs = fc_layer(x)
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
        