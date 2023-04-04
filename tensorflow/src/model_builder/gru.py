import tensorflow as tf


class GRU(tf.keras.Model):
    def __init___(self, cfg):
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
            self.flag_use_gru_blocks = False
            
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(cfg.UNITS),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED),
            tf.keras.layers.Dense(cfg.UNITS//2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED),
            tf.keras.layers.Dense(cfg.NUM_CLASSES)
        ])

    def call(self, x):
        x = self.start_gru(x)
        if self.flag_use_gru_block:
            for blk in self.gru_blocks:
                x = blk(x)
        x = self.end_gru(x)
        x = self.mlp(x)
        return x

        