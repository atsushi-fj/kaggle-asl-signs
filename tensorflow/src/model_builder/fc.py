import tensorflow as tf


class FC(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()        
        self.end_fc = tf.keras.layers.Dense(cfg.NUM_CLASSES,
                                            activation="softmax")
        
        if (cfg.NUM_BLOCKS - 1) > 0:
            self.fc_blocks = [
                tf.keras.layers.Dense(cfg.UNITS),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(cfg.MLP_DROPOUT_RATE, seed=cfg.SEED),] * (cfg.NUM_BLOCKS - 1)
            self.flag_use_fc_block = True
        else:
            self.flag_use_fc_block = False
            
    def get_config(self):
        config = {
            "end_fc" : self.end_fc,
            "flag_use_fc_block" : self.flag_use_fc_block
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        if self.flag_use_fc_block:
            for blk in self.fc_blocks:
                x = blk(x)
        x = self.end_fc(x)
        return x