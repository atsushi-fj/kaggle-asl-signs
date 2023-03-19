import numpy as np
import torch
import wandb
from pathlib import Path

from .model_builder.backborn.baseline import Baseline
from .engine import train
from .utils import load_config, EarlyStopping, seed_everything, create_display_name
from .data_loader import data_split, create_dataloader
from .inference import eval_model, accuracy_fn


def run(project,
        experiment_name,
        model_name,
        extra=None,
        config="config.yaml"):
    
    name = create_display_name(experiment_name=experiment_name,
                               model_name=model_name,
                               extra=extra)
    cfg = load_config(file=config)
    
    with wandb.init(project=project,
                    name=name,
                    config=cfg):
        cfg = wandb.config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        
        seed_everything(seed=cfg.seed)

        # load feature data
        data_path = Path("./input/processed_data")
        X = np.load(data_path / "X.npy")
        y = np.load(data_path / "y.npy")

        # Split data
        train_dataset, val_dataset = data_split(X, y,
                                                test_size=cfg.test_size,
                                                random_state=cfg.seed,
                                                stratify=y)

        # Create dataloader
        train_dataloader, val_dataloader = create_dataloader(train_dataset, val_dataset,
                                                            batch_size=cfg.batch_size,
                                                            pin_memory=True,
                                                            train_drop_last=True)
        # Training model
        model = Baseline().to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        earlystopping = EarlyStopping(patience=cfg.patience, verbose=True)
        
        
        train(model, train_dataloader, val_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochs=cfg.epochs,
            earlystopping=earlystopping,
            model_name=cfg.model_name,
            device=device)
        
        model.load_state_dict(torch.load(f=cfg.load_model_path))
        model.to(device)
        
        result = eval_model(model=model,
                            data_loader=val_dataloader,
                            loss_fn=loss_fn,
                            accuracy_fn=accuracy_fn,
                            device=device)
        
        print(f"\n{result}")

