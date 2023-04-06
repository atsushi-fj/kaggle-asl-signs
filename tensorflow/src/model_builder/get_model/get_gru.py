import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from ...utils import load_gru_data, get_lr_metric
from ...feature import create_gru_features
from ..gru import GRU


def get_gru(cfg):
    X, y = load_gru_data()
    X = create_gru_features(X)
    print(X.shape)
    
    inputs = tf.keras.layers.Input(shape=(cfg.BATCH_SIZE, X.shape[1], cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    outputs = GRU(cfg)(inputs)
    
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
