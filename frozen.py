#unused file
import tensorflow as tf
import tf2onnx
import onnx

model = "/home/sonnet/Downloads/Traffic_Signal_Sonnet/output/Traffic_Light_classifier/saved_model.pb"

input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)
onnx.save(onnx_model, "/home/sonnet/Downloads/Traffic_Signal_Sonnet/output/model.onnx")
