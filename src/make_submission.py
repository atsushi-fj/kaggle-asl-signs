import torch
import tensorflow as tf
import subprocess
import onnx
import onnx_tf
from onnx_tf.backend import prepare
from .utils import load_config


def make_submission(model,
                    model_path,
                    config,
                    device,
                    onnx_model_path="model.onnx",
                    tf_model_path="model_tf",
                    tflite_model_path="model.tflite"):
    
    cfg = load_config(file=config)
    model.load_state_dict(torch.load(model_path), strict=False).to(device)
    model.eval()
    sample_input = torch.rand((cfg.batch_size, *cfg.img_size)).to(device)
    torch.onnx.export(
        model,
        sample_input,
        onnx_model_path,
        verbose=False,
        input_names=["inputs"],
        output_names=["outputs"],
        opset_version=12
    )
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_model_path)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    tflite_model = converter.convert()
    
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
        
