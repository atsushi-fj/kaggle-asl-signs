import tensorflow as tf
import pandas as pd
from .preprocess import PreprocessLayer
from .utils import load_relevant_data_subset, load_config
from .model_builder.final_model.transformer import FinalTransformer


def submit(model_path, config):
    cfg = load_config(config)
    train = pd.read_csv(cfg.TRAIN_CSV_PATH)
    demo_raw_data = load_relevant_data_subset(train['file_path'].values[1])
    model = tf.keras.models.load_model(model_path)
    preprocess_layer = PreprocessLayer(cfg)
    final_model = FinalTransformer(model, preprocess_layer)
    demo_output = final_model(demo_raw_data)["outputs"]
    demo_prediction = demo_output.numpy().argmax()
    print(f'demo_prediction: {demo_prediction}, correct: {train.iloc[1]["sign_ord"]}')
    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('./model.tflite', 'wb') as f:
        f.write(tflite_model)
    