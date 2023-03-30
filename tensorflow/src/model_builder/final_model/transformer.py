import tensorflow as tf


class FinalTransformer(tf.keras.Model):
    def __init__(self, model, pp_layer):
        super().__init__()
        self.model = model
        self.pp_layer = pp_layer


    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 543, 3],
                                                dtype=tf.float32,
                                                name='inputs')])        
    def __call__(self, inputs):
        x, non_empty_frame_idxs = self.pp_layer(inputs)
        x = tf.expand_dims(x, axis=0)
        non_empty_frame_idxs = tf.expand_dims(non_empty_frame_idxs, axis=0)
        outputs = self.model({ 'frames': x, 'non_empty_frame_idxs': non_empty_frame_idxs })
        outputs = tf.squeeze(outputs, axis=0)
        # Return a dictionary with the output tensor
        return {'outputs': outputs}
    