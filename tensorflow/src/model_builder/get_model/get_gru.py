import tensorflow as tf
import tensorflow_addons as tfa
from ...utils import load_data, get_lr_metric
from ..gru import GRU


def get_gru(cfg):
    X, y, _ = load_data()
    
    INPUT_SHAPE = (cfg.NUM_FRAMES, cfg.N_COLS * (cfg.N_DIMS-1))
    NUM_BASE_FEATS = (cfg.SEGMENTS + 1) * INPUT_SHAPE[1] * 2
    FLAT_FRAME_SHAPE = NUM_BASE_FEATS + (INPUT_SHAPE[0] * INPUT_SHAPE[1])
    
    # inputs = tf.keras.layers.Input(shape=(FLAT_FRAME_SHAPE,),
    #                                dtype=tf.float32,
    #                                name="inputs")
    # x = inputs[:, :NUM_BASE_FEATS]
    # x = tf.reshape(inputs[:, NUM_BASE_FEATS:], (-1, cfg.NUM_FRAMES, INPUT_SHAPE[1]))
    inputs = tf.keras.layers.Input(shape=(-1, cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS),
                                   dtype=tf.float32,
                                   name="inputs")
    x = tf.reshape(inputs, (-1, cfg.INPUT_SIZE, cfg.N_COLS * cfg.N_DIMS))
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
