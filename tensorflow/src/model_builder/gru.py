import tensorflow as tf


class GRU(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.start_gru = tf.keras.layers.GRU(cfg.UNITS,
                                            dropout=0.0,
                                            return_sequences=True)
        
        self.end_gru = tf.keras.layers.GRU(cfg.UNITS,
                                           dropout=cfg.DROPRATE,
                                           return_sequences=False)
        if (cfg.NUM_BLOCKS - 2) > 0:
            self.gru_blocks = [
                tf.keras.layers.GRU(cfg.UNITS,
                                    dropout=cfg.DROPRATE,
                                    return_sequences=True)] * (cfg.NUM_BLOCKS - 2)
            self.flag_use_gru_block = True
        else:
            self.flag_use_gru_block = False
            
        self.mlp = tf.keras.Sequential([
            # tf.keras.layers.Dense(cfg.UNITS),
            # tf.keras.layers.BatchNormalization(),
            # tf.keras.layers.ReLU(),
            # tf.keras.layers.Dropout(cfg.MLP_DROPOUT_RATE, seed=cfg.SEED),
            tf.keras.layers.Dense(cfg.UNITS),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(cfg.MLP_DROPOUT_RATE, seed=cfg.SEED),
            tf.keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax")
        ])

    def call(self, x):
        x = self.start_gru(x)
        if self.flag_use_gru_block:
            for blk in self.gru_blocks:
                x = blk(x)
        x = self.end_gru(x)
        x = self.mlp(x)
        return x


def gru_block(inputs, cfg):
    vector = tf.keras.layers.GRU(cfg.UNITS,
                                 dropout=cfg.DROPRATE,
                                 return_sequences=True,)(inputs)
    vector = tf.keras.layers.GRU(cfg.UNITS,
                                 dropout=cfg.DROPRATE,
                                 return_sequences=False,)(vector)
    vector = tf.keras.layers.Dropout(0.3)(vector)
    return vector


def mlp_block(inputs, cfg):
    x = tf.keras.layers.Dense(cfg.MLP_UNITS)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(cfg.MLP_DROPOUT_RATE, seed=cfg.SEED)(x)
    x = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax")(x)
    return x
