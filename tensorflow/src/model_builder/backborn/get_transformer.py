import tensorflow as tf
import tensorflow_addons as tfa
from utils import load_config, get_lr_metric
from transformer.embedding import CustomEmbedding
from transformer.transformer import Transformer

def get_model(cfg):
    # Inputs
    frames = tf.keras.layers.Input([cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([cfg.INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)
    
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,cfg.INPUT_SIZE, cfg.N_COLS, 2])
    # LIPS
    lips = tf.slice(x, [0,0,cfg.LIPS_START,0], [-1,cfg.INPUT_SIZE, 40, 2])
    lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - cfg.LIPS_MEAN) / cfg.LIPS_STD,
        )
    lips = tf.reshape(lips, [-1, cfg.INPUT_SIZE, 40*2])
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,40,0], [-1,cfg.INPUT_SIZE, 21, 2])
    left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - cfg.LEFT_HANDS_MEAN) / cfg.LEFT_HANDS_STD,
        )
    left_hand = tf.reshape(left_hand, [-1, cfg.INPUT_SIZE, 21*2])
    # RIGHT HAND
    right_hand = tf.slice(x, [0,0,61,0], [-1,cfg.INPUT_SIZE, 21, 2])
    right_hand = tf.where(
            tf.math.equal(right_hand, 0.0),
            0.0,
            (right_hand - cfg.RIGHT_HANDS_MEAN) / cfg.RIGHT_HANDS_STD,
        )
    right_hand = tf.reshape(right_hand, [-1, cfg.INPUT_SIZE, 21*2])
    # POSE
    pose = tf.slice(x, [0,0,82,0], [-1,cfg.INPUT_SIZE, 10, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - cfg.POSE_MEAN) / cfg.POSE_STD,
        )
    pose = tf.reshape(pose, [-1, cfg.INPUT_SIZE, 10*2])
    x = lips, left_hand, right_hand, pose
    x = CustomEmbedding()(lips, left_hand, right_hand, pose, non_empty_frame_idxs)
    # Encoder Transformer Blocks
    x = Transformer(cfg.NUM_BLOCKS)(x, mask)
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classification Layer
    x = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation=tf.keras.activations.softmax, kernel_initializer=INIT_GLOROT_UNIFORM)(x)
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    
    # Simple Categorical Crossentropy Loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    
    # Adam Optimizer with weight decay
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    
    lr_metric = get_lr_metric(optimizer)
    metrics = ["acc",lr_metric]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model