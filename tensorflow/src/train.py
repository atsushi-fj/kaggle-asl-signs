import tensorflow as tf
from .utils import WeightDecayCallback, load_config, lrfn, \
                   create_kfold, create_display_name
from .model_builder import get_gru, get_transformer
from .train_generator import get_train_batch_all_signs_gru, get_train_batch_all_signs
import wandb


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
                                             patience=30),
            wandb.keras.WandbCallback()]
        
        print("traing 開始")
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
    artifact = wandb.Artifact(cfg.ARTIFACT, type='model')
    artifact.add_file(cfg.ARTIFACT_DATA)
    artifact.add_file(cfg.ARTIFACT_INDEX)
    run.log_artifact(artifact)


def run_gru(config):
    cfg = load_config(config)
    X, y, model = get_gru(cfg)
    
    if cfg.CREATE_KFOLD:
        train_idxs, val_idxs = create_kfold(cfg)
        X_train = X[train_idxs]
        X_val = X[val_idxs]
        y_train = y[train_idxs]
        y_val = y[val_idxs]
        
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
                                             patience=30),
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
            x=get_train_batch_all_signs_gru(X, y, cfg),
            steps_per_epoch=len(X) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
            epochs=cfg.N_EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=2,)
     
    model.save(cfg.MODEL_PATH)
    model.save_weights(cfg.MODEL_WEIGHTS_PATH)
    artifact = wandb.Artifact(cfg.ARTIFACT, type='model')
    artifact.add_file(cfg.ARTIFACT_DATA)
    artifact.add_file(cfg.ARTIFACT_INDEX)
    run.log_artifact(artifact)
