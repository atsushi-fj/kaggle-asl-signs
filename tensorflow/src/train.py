import tensorflow as tf
from .utils import WeightDecayCallback, load_config, lrfn, get_train_batch_all_signs
from .model_builder import get_transformer
import wandb
import numpy as np


def run(config):
    cfg = load_config(config)
    X, y, NON_EMPTY_FRAME_IDXS, model = get_transformer()
    
    run = wandb.init(project="kaggle-asl-signs", config=cfg, tags=['transformer', 'final-model'])
    tf.keras.backend.clear_session()
    # Learning rate for encoder
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=cfg.N_WARMUP_EPOCHS, lr_max=cfg.LR_MAX, num_cycles=0.50) for step in range(cfg.N_EPOCHS)]
    
    # Learning Rate Callback
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    callbacks=[
            lr_callback,
            WeightDecayCallback(),
            wandb.keras.WandbCallback()
    ]
    
    model.fit(
        x=get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS),
        steps_per_epoch=len(X) // (cfg.NUM_CLASSES * cfg.BATCH_ALL_SIGNS_N),
        epochs=cfg.N_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,)
     
    model.save('./final_model_one')
    model.save_weights('./final_model_one_weights')
    artifact = wandb.Artifact('final_model_one', type='model')
    artifact.add_file('./final_model_one_weights.data-00000-of-00001')
    artifact.add_file('./final_model_one_weights.index')
    run.log_artifact(artifact)
    