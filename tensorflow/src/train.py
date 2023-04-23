import tensorflow as tf
from .utils import WeightDecayCallback, load_config, lrfn, \
                   create_kfold, create_display_name
from .model_builder import get_gru, get_transformer, get_feature_gru, get_new_feature_gru, get_residual_gru, get_fc
from .train_generator import get_train_batch_all_signs, get_gru_dataset_kfold, get_gru_dataset_not_kfold, get_train_batch_all_signs_ln, get_train_batch_all_signs_gru
import wandb
import numpy as np


def run_transformer(config):
    cfg = load_config(config)
    X, y, NON_EMPTY_FRAME_IDXS, model = get_transformer(cfg)
    
    if cfg.CREATE_KFOLD:
        train_idxs, val_idxs = create_kfold(cfg)
        X_train = X[train_idxs]
        X_val = X[val_idxs]
        NON_EMPTY_FRAME_IDXS_TRAIN = NON_EMPTY_FRAME_IDXS[train_idxs]
        NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS[val_idxs]
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        print(f'# NaN Values In Prediction: {np.isnan(X_train).sum()}')
        
    name = create_display_name(experiment_name=cfg.EXPERIMENT_NAME,
                               model_name=cfg.MODEL_NAME)
    run = wandb.init(project=cfg.PROJECT,
                     name=name,
                     config=cfg)
    # Learning rate for encoder
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.N_WARMUP_EPOCHS,
                        lr_max=cfg.LR_MAX,
                        cfg=cfg,
                        num_cycles=0.50) for step in range(cfg.N_EPOCHS)]
    
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    
    if cfg.CREATE_KFOLD:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=cfg.PATIENCE,
                                             restore_best_weights=True),
            wandb.keras.WandbCallback()]
        model.fit(
        x=get_train_batch_all_signs(X_train,
                                    y_train,
                                    NON_EMPTY_FRAME_IDXS_TRAIN,
                                    cfg),
        steps_per_epoch=len(X_train) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        validation_data=get_train_batch_all_signs(X_val,
                                                  y_val,
                                                  NON_EMPTY_FRAME_IDXS_VAL,
                                                  cfg),
        validation_steps=len(X_val) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        epochs=cfg.N_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,)
        
    else:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            wandb.keras.WandbCallback()]
        
        model.fit(
            x=get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, cfg),
            steps_per_epoch=len(X) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
            epochs=cfg.N_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,)
     
    model.save(cfg.MODEL_PATH)
    model.save_weights(cfg.MODEL_WEIGHTS_PATH)


def run_gru(config):
    cfg = load_config(config)
    X, y, model = get_residual_gru(cfg)
    
    # create dataset
    if cfg.CREATE_KFOLD:
        train_idxs, val_idxs = create_kfold(cfg)
        X_train = X[train_idxs]
        X_val = X[val_idxs]
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        train_dataset, val_dataset = get_gru_dataset_kfold(batch_size=cfg.BATCH_SIZE,
                                                           X_train=X_train,
                                                           y_train=y_train,
                                                           X_val=X_val,
                                                           y_val=y_val)
    else:
        train_dataset = get_gru_dataset_not_kfold(batch_size=cfg.BATCH_SIZE,
                                                  X_train=X,
                                                  y_train=y)
        
    name = create_display_name(experiment_name=cfg.EXPERIMENT_NAME,
                               model_name=cfg.MODEL_NAME)
    run = wandb.init(project=cfg.PROJECT,
                     name=name,
                     config=cfg)
    
    # Learning rate for encoder
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.N_WARMUP_EPOCHS,
                        lr_max=cfg.LR_MAX,
                        cfg=cfg,
                        num_cycles=0.50) for step in range(cfg.N_EPOCHS)]
    
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    
    if cfg.CREATE_KFOLD:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=cfg.PATIENCE,
                                             restore_best_weights=True),
            wandb.keras.WandbCallback()]
        
        model.fit(x=train_dataset,
                  steps_per_epoch=len(train_dataset),
                  validation_data=val_dataset,
                  validation_steps=len(val_dataset),
                  epochs=cfg.N_EPOCHS,
                  batch_size=cfg.BATCH_SIZE,
                  callbacks=callbacks,
                  verbose=2)
        
    else:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            wandb.keras.WandbCallback()]
        
        model.fit(
            x=train_dataset,
            steps_per_epoch=len(train_dataset),
            epochs=cfg.N_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2)
     
    model.save(cfg.MODEL_PATH)
    model.save_weights(cfg.MODEL_WEIGHTS_PATH)


def run_fc(config):
    cfg = load_config(config)
    X, y, model = get_fc(cfg)
    
    if cfg.CREATE_KFOLD:
        train_idxs, val_idxs = create_kfold(cfg)
        X_train = X[train_idxs]
        X_val = X[val_idxs]
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        print(f'# NaN Values In Prediction: {np.isnan(X_train).sum()}')
        
    name = create_display_name(experiment_name=cfg.EXPERIMENT_NAME,
                               model_name=cfg.MODEL_NAME)
    run = wandb.init(project=cfg.PROJECT,
                     name=name,
                     config=cfg)
    # Learning rate for encoder
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.N_WARMUP_EPOCHS,
                        lr_max=cfg.LR_MAX,
                        cfg=cfg,
                        num_cycles=0.50) for step in range(cfg.N_EPOCHS)]
    
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    
    if cfg.CREATE_KFOLD:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=cfg.PATIENCE,
                                             restore_best_weights=True),
            wandb.keras.WandbCallback()]
        model.fit(
        x=get_train_batch_all_signs_ln(X_train,
                                    y_train,
                                    cfg),
        steps_per_epoch=len(X_train) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        validation_data=get_train_batch_all_signs_ln(X_val,
                                                  y_val,
                                                  cfg),
        validation_steps=len(X_val) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        epochs=cfg.N_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,)
        
    else:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            wandb.keras.WandbCallback()]
        
        model.fit(
            x=get_train_batch_all_signs_ln(X, y, cfg),
            steps_per_epoch=len(X) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
            epochs=cfg.N_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,)
     
    model.save(cfg.MODEL_PATH)
    model.save_weights(cfg.MODEL_WEIGHTS_PATH)
    
    
def run_gru2(config):
    cfg = load_config(config)
    X, y, model = get_residual_gru(cfg)
    
    if cfg.CREATE_KFOLD:
        train_idxs, val_idxs = create_kfold(cfg)
        X_train = X[train_idxs]
        X_val = X[val_idxs]
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        print(f'# NaN Values In Prediction: {np.isnan(X_train).sum()}')
        
    name = create_display_name(experiment_name=cfg.EXPERIMENT_NAME,
                               model_name=cfg.MODEL_NAME)
    run = wandb.init(project=cfg.PROJECT,
                     name=name,
                     config=cfg)
    # Learning rate for encoder
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.N_WARMUP_EPOCHS,
                        lr_max=cfg.LR_MAX,
                        cfg=cfg,
                        num_cycles=0.50) for step in range(cfg.N_EPOCHS)]
    
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    
    if cfg.CREATE_KFOLD:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=cfg.PATIENCE,
                                             restore_best_weights=True),
            wandb.keras.WandbCallback()]
        model.fit(
        x=get_train_batch_all_signs_gru(X_train,
                                    y_train,
                                    cfg),
        steps_per_epoch=len(X_train) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        validation_data=get_train_batch_all_signs_gru(X_val,
                                                  y_val,
                                                  cfg),
        validation_steps=len(X_val) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        epochs=cfg.N_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,)
        
    else:
        callbacks=[
            lr_callback,
            WeightDecayCallback(model=model, cfg=cfg),
            wandb.keras.WandbCallback()]
        
        model.fit(
            x=get_train_batch_all_signs(X, y, cfg),
            steps_per_epoch=len(X) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
            epochs=cfg.N_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,)
     
    model.save(cfg.MODEL_PATH)
    model.save_weights(cfg.MODEL_WEIGHTS_PATH)


