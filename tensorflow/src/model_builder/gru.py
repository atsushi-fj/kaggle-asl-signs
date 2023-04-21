import tensorflow as tf

output_bias = tf.keras.initializers.Constant(1.0 / 250.0)

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


class GRUOnly(tf.keras.Model):
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

    def call(self, x):
        x = self.start_gru(x)
        if self.flag_use_gru_block:
            for blk in self.gru_blocks:
                x = blk(x)
        x = self.end_gru(x)
        return x



class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, droprate, cfg):
        super().__init__()
        self.linear = tf.keras.layers.Dense(cfg.RESIDUAL_UNITS)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("gelu")
        if droprate != 0:
            self.drop = tf.keras.layers.Dropout(droprate)
            self.flag_use_drop = True
        else:
            self.flag_use_drop = False

    def call(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        if self.flag_use_drop:
            x = self.drop(x)
        return x

    
class MSD(tf.keras.layers.Layer):
    def __init__(
        self,
        cfg,
        fold_num=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.lin = tf.keras.layers.Dense(
            cfg.NUM_CLASSES,
            activation=None,
            use_bias=True,
            bias_initializer=output_bias,
            # kernel_regularizer=R.l2(WEIGHT_REGULARIZE)
        )

        rate_dropout = 0.5
        self.dropouts = [
            tf.keras.layers.Dropout((rate_dropout - 0.2), seed=135 + fold_num),
            tf.keras.layers.Dropout((rate_dropout - 0.1), seed=690 + fold_num),
            tf.keras.layers.Dropout((rate_dropout), seed=275 + fold_num),
            tf.keras.layers.Dropout((rate_dropout + 0.1), seed=348 + fold_num),
            tf.keras.layers.Dropout((rate_dropout + 0.2), seed=861 + fold_num),
        ]
        
    def call(self, inputs):
        for ii, drop in enumerate(self.dropouts):
            if ii == 0:
                out = self.lin(drop(inputs)) / 5.0
            else:
                out += self.lin(drop(inputs)) / 5.0
        return out
        

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
