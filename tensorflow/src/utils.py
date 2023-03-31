from pathlib import Path
import yaml
import tensorflow as tf
import numpy as np
import math
import pandas as pd
from argparse import Namespace
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, cfg):
        self.model = model
        self.step_counter = 0
        self.wd_ratio = cfg.WD_RATIO
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')


def lrfn(current_step, num_warmup_steps, lr_max, cfg, num_cycles=0.50):
    num_training_steps=cfg.N_EPOCHS
    
    if current_step < num_warmup_steps:
        if WARMUP_METHOD == 'log':
            return lr_max * 0.10 ** (num_warmup_steps - current_step)
        else:
            return lr_max * 2 ** -(num_warmup_steps - current_step)
    else:
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max
    

def load_config(file="config.yaml"):
    """Load config file"""
    config_path = Path("./config/")
    with open(config_path / file, 'r') as file:
        cfg = yaml.safe_load(file)
    cfg = Namespace(**cfg)
    return cfg


def load_data():
    X = np.load('input/X.npy')
    y = np.load('input/y.npy')
    NON_EMPTY_FRAME_IDXS = np.load('input/NON_EMPTY_FRAME_IDXS.npy')
    return X, y, NON_EMPTY_FRAME_IDXS 


# Custom sampler to get a batch containing N times all signs
def get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, cfg):
    n=cfg.BATCH_ALL_SIGNS_N
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
        
        yield { 'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch }, y_batch


def create_kfold(cfg):
    train = pd.read_csv(cfg.TRAIN_CSV_PATH)
    sgkf = StratifiedGroupKFold(n_splits=cfg.k, shuffle=True, random_state=cfg.SEED)
    train['fold'] = -1
    for i, (train_idx, val_idx) in enumerate(sgkf.split(train.index, train.sign, train.participant_id)):
        train.loc[val_idx, 'fold'] = i 
    train_idxs = train.query("fold!=0").index.values
    val_idxs = train.query("fold==0").index.values
    return train_idxs, val_idxs


def load_relevant_data_subset(pq_path, cfg):
    data_columns = cfg.DIM_NAMES
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / cfg.N_ROWS)
    data = data.values.reshape(n_frames, cfg.N_ROWS, len(data_columns))
    return data.astype(np.float32)


def create_display_name(experiment_name,
                        model_name,
                        extra=None):

    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        name = f"{timestamp}-{experiment_name}-{model_name}-{extra}"
    else:
        name = f"{timestamp}-{experiment_name}-{model_name}"
    print(f"[INFO] Create wandb saving to {name}")
    return name