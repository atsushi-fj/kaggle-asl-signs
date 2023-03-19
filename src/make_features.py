import os
import numpy as np
import tqdm
import torch
from torch import nn
import pandas as pd
import multiprocessing as mp

from utils import load_config

# Load config file 
cfg = load_config(file="config.yaml")

class FeatureGen(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        x = x[:, :, :2]
        lips_x = x[:, cfg.lips, :].contiguous().view(-1, 43 * 2)
        lefth_x = x[:, 468:489, :].contiguous().view(-1, 21 * 2)
        pose_x = x[:, 489:522, :].contiguous().view(-1, 33 * 2)
        righth_x = x[:, 522:, :].contiguous().view(-1, 21 * 2)

        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1), :]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1), :]

        x1m = torch.mean(lips_x, 0)
        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)

        x1s = torch.std(lips_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)

        xfeat = torch.cat([x1m, x2m, x3m, x4m, x1s, x2s, x3s, x4s], axis=0)
        xfeat = torch.where(
            torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat
        )


def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / cfg.rows_per_frame)
    data = data.values.reshape(n_frames, cfg.rows_per_frame, len(data_columns))
    return data.astype(np.float32)


def convert_row(row):
    x = load_relevant_data_subset(os.path.join("/kaggle/input/asl-signs", row[1].path))
    x = FeatureGen(torch.tensor(x)).cpu().numpy()
    return x, row[1].label


def convert_and_save_data():
    df = pd.read_csv(TRAIN_FILE)  # train.csv
    df['label'] = df['sign'].map(label_map)  # signを数字でラベリングした行を追加
    npdata = np.zeros((df.shape[0], 472))  # それぞれ特徴量の配列を形だけ作っておく
    nplabels = np.zeros(df.shape[0])
    with mp.Pool() as pool:
        results = pool.imap(convert_row, df.iterrows(), chunksize=250)
        for i, (x,y) in tqdm(enumerate(results), total=df.shape[0]):
            npdata[i,:] = x
            nplabels[i] = y
    
    np.save("../input/processed_data/feature_data.npy", npdata)
    np.save("../input/processed_data/feature_labels.npy", nplabels)


if __name__ == "__main__":
    convert_and_save_data()
