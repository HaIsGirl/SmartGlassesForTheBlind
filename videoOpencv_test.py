import tflite_runtime.interpreter as tflite
import numpy as np
import cv2

# Load mô hình TensorFlow Lite
interpreter = tflite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Lấy thông tin đầu vào và đầu ra của mô hình
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(image_path):
    # Đọc và chuẩn bị ảnh đầu vào
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)  # Batch size

    # Cấp phát bộ nhớ cho đầu vào
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # Lấy kết quả dự đoán
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Predicted output:", output_data)
    return np.argmax(output_data)


