import tensorflow as tf
import tf_keras
# Bước 1: Tải mô hình từ tệp .h5
model = tf_keras.models.load_model("/home/haisgirl/Documents/mycode/Python/SmartGlassesForTheBlind/models/model.h5")

# Bước 2: Chuyển đổi mô hình sang định dạng TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Bước 3: Lưu mô hình dưới định dạng .tflite
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Chuyển đổi thành công mô hình sang TFLite!")
