import tensorflow as tf
import numpy as np


def get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, cfg):
    """Custom sampler to get a batch containing N times all signs"""
    n = cfg.BATCH_ALL_SIGNS_N
    # Arrays to store batch in
    X_batch = np.zeros([cfg.NUM_CLASSES*n, cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, cfg.NUM_CLASSES, step=1/n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([cfg.NUM_CLASSES*n, cfg.INPUT_SIZE], dtype=np.float32)
    
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(cfg.NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
            
    while True:
        # Fill batch arrays
        for i in range(cfg.NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i*n:(i+1)*n] = X[idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n] = NON_EMPTY_FRAME_IDXS[idxs]
        
        yield {'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch}, y_batch


def get_train_batch_all_signs_gru(X, y, cfg):
    """Custom sampler to get a batch containing N times all signs for GRU"""
    n = cfg.BATCH_ALL_SIGNS_N
    # Arrays to store batch in
    X_batch = np.zeros([cfg.NUM_CLASSES*n, cfg.INPUT_SIZE, cfg.N_COLS, cfg.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, cfg.NUM_CLASSES, step=1/n, dtype=np.float32).astype(np.int64)
    
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(cfg.NUM_CLASSES):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
            
    while True:
        # Fill batch arrays
        for i in range(cfg.NUM_CLASSES):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i*n:(i+1)*n] = X[idxs]
        
        yield X_batch, y_batch


def get_gru_dataset_kfold(batch_size, X_train, y_train, X_val, y_val):
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_features_dataet = tf.data.Dataset.from_tensor_slices(X_val)
    val_labels_dataset = tf.data.Dataset.from_tensor_slices(y_val)
    val_dataset = tf.data.Dataset.zip((val_features_dataet, val_labels_dataset))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    return train_dataset, val_dataset


def get_gru_dataset_not_kfold(batch_size, X_train, y_train):
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
    return train_dataset

