from pathlib import Path
import yaml
import tensorflow as tf
import numpy as np
import math
import pandas as pd
from argparse import Namespace
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold


class WeightDecayCallback(tf.keras.callbacks.Callback):
    """Custom callback to update weight decay with learning rate"""
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


def load_gru_data():
    X = np.load("input/X_gru.npy")
    y = np.load("input/y_gru.npy")
    return X, y


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


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr