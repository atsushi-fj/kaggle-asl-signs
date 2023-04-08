import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from ...utils import load_data, get_lr_metric
from ..gru import GRU


def get_gru(cfg):
    X, y, _ = load_data()
    
    inputs = tf.keras.layers.Input(shape=(cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    x = tf.reshape(inputs, [-1, cfg.INPUT_SIZE, cfg.N_COLS * cfg.N_DIMS])
    outputs = GRU(cfg)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    loss = "sparse_categorical_crossentropy"
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     clipnorm=cfg.clipnorm)
    
    lr_metric = get_lr_metric(optimizer)
    metrics = ["acc",lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return X, y, model
