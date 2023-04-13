import tensorflow as tf
import tensorflow_addons as tfa
from ..transformer import CustomEmbedding
from ..transformer import Transformer
from ...utils import load_data, load_input64_data
from ...feature import create_feature_statistics_input64


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


def scce_with_ls(y_true, y_pred):
    # One Hot Encode Sparsely Encoded Target Sign
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, 250, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    # Categorical Crossentropy with native label smoothing support
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)


def get_transformer(cfg):
    X, y, NON_EMPTY_FRAME_IDXS = load_input64_data()
    feature_stats = create_feature_statistics_input64(X)
    # Inputs
    frames = tf.keras.layers.Input([cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS],
                                   dtype=tf.float32,
                                   name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([cfg.INPUT_SIZE],
                                                 dtype=tf.float32,
                                                 name='non_empty_frame_idxs')
    # Padding Mask
    mask0 = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask0 = tf.expand_dims(mask0, axis=2)
    # Random Frame Masking
    mask = tf.where(
        (tf.random.uniform(tf.shape(mask0)) > 0.25) & tf.math.not_equal(mask0, 0.0),
        1.0,
        0.0,
    )
    # Correct Samples Which are all masked now...
    mask = tf.where(
        tf.math.equal(tf.reduce_sum(mask, axis=[1,2], keepdims=True), 0.0),
        mask0,
        mask,
    )
    
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,cfg.INPUT_SIZE, cfg.N_COLS, 2])
    
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
    pose = tf.slice(x, [0,0,82,0], [-1,cfg.INPUT_SIZE, 10, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - feature_stats["pose"][1]) / feature_stats["pose"][2],
        )
    pose = tf.reshape(pose, [-1, cfg.INPUT_SIZE, 10*2])
    

    x = CustomEmbedding(cfg)(lips, left_hand, pose, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    x = Transformer(cfg.NUM_BLOCKS, cfg)(x, mask)
    
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    
    # Classifier Dropout
    x = tf.keras.layers.Dropout(cfg.CLASSIFIER_DROPOUT_RATIO)(x)
    
    # Classification Layer
    x = tf.keras.layers.Dense(cfg.NUM_CLASSES,
                              activation=tf.keras.activations.softmax,
                              kernel_initializer=tf.keras.initializers.glorot_uniform)(x)
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Sparse Categorical Cross Entropy With Label Smoothing
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=cfg.lr,
                                     weight_decay=cfg.weight_decay,
                                     clipnorm=cfg.clipnorm)
    
    lr_metric = get_lr_metric(optimizer)
    
    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
        lr_metric
    ]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return X, y, NON_EMPTY_FRAME_IDXS, model
