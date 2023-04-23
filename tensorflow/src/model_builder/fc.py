import tensorflow as tf


class FC(tf.keras.Model):
    def __init__(self, cfg):
        super().__init__()
        self.start_fc = tf.keras.Sequential([
            tf.keras.layers.Dense(cfg.UNITS),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED)
        ])
                
        self.end_fc = tf.keras.layers.Dense(cfg.NUM_CLASSES,
                                            activation="softmax")
        
        if (cfg.NUM_BLOCKS - 2) > 0:
            self.fc_blocks = [
                tf.keras.layers.Dense(cfg.UNITS),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(cfg.DROPRATE, seed=cfg.SEED)] * (cfg.NUM_BLOCKS - 2)
            self.flag_use_fc_block = True
        else:
            self.flag_use_fc_block = False
            
    def get_config(self):
        config = {
            "start_fc": self.start_fc,
            "end_fc" : self.end_fc,
            "flag_use_fc_block" : self.flag_use_fc_block
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x):
        x = self.start_fc(x)
        if self.flag_use_fc_block:
            for blk in self.fc_blocks:
                x = blk(x)
        x = self.end_fc(x)
        return x