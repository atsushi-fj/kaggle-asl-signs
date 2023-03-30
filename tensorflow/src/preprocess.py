import tensorflow as tf
import numpy as np


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)
    
    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, 543, 3], dtype=tf.float32),),
    )
    def call(self, data0):
        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]
        LIPS_IDXS0 = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])
        LEFT_HAND_IDXS0  = np.arange(468,489)
        RIGHT_HAND_IDXS0 = np.arange(522,543)
        POSE_IDXS0       = np.arange(502, 512)
        LANDMARK_IDXS0   = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0, POSE_IDXS0))
        HAND_IDXS0       = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
        
        # Keep only non-empty frames in data
        frames_hands_nansum = tf.experimental.numpy.nanmean(tf.gather(data0, HAND_IDXS0, axis=1), axis=[1,2])
        non_empty_frames_idxs = tf.where(frames_hands_nansum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)
        
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32) 
        
        N_FRAMES = tf.shape(data)[0]
        data = tf.gather(data, LANDMARK_IDXS0, axis=1)
        
        if N_FRAMES < self.cfg.INPUT_SIZE:
            # Video fits in cfg.INPUT_SIZE
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, self.cfg.INPUT_SIZE-N_FRAMES]], constant_values=-1)
            data = tf.pad(data, [[0, self.cfg.INPUT_SIZE-N_FRAMES], [0,0], [0,0]], constant_values=0)
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        else:
            # Video needs to be downsampled to cfg.INPUT_SIZE
            if N_FRAMES < self.cfg.INPUT_SIZE**2:
                repeats = tf.math.floordiv(self.cfg.INPUT_SIZE * self.cfg.INPUT_SIZE, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)

            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), self.cfg.INPUT_SIZE)
            if tf.math.mod(len(data), self.cfg.INPUT_SIZE) > 0:
                pool_size += 1
            if pool_size == 1:
                pad_size = (pool_size * self.cfg.INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * self.cfg.INPUT_SIZE) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(self.cfg.INPUT_SIZE, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(self.cfg.INPUT_SIZE, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            data = tf.reshape(data, [self.cfg.INPUT_SIZE, -1, self.cfg.N_COLS, self.cfg.N_DIMS])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [self.cfg.INPUT_SIZE, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)

            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            
            return data, non_empty_frames_idxs
    
