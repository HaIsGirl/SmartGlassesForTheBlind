import numpy as np
import cv2
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = None
detections = None

def loadModel(prototxt, model):
	global net
	print("Loading model...")
	net = cv2.dnn.readNetFromCaffe(prototxt, model)

def process(frame):
	global net, detections
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                  0.007843, (300, 300), 127.5)
                                  
	net.setInput(blob)
	detections = net.forward()
	return detections

def predict(detections, confidence_threshold):
	for i in np.arange(0, detections.shape[2]):
		# Get the confidence
		confidence = detections[0, 0, i, 2]
		if confidence > confidence_threshold:
			value = int(detections[0, 0, i, 1])
			label = "{}: {:.2f}%".format(CLASSES[value], confidence * 100)
			print(label)



def drawOnImage(frame, detections, confidence_threshold):
    """
    Vẽ khung và nhãn lên hình ảnh dựa trên kết quả phát hiện.

    Args:
        frame (numpy.ndarray): Khung hình (ảnh).
        detections (numpy.ndarray): Ma trận chứa thông tin phát hiện (định dạng [1, 1, N, 7]).
        confidence_threshold (float): Ngưỡng tự tin để vẽ khung (mặc định: 0.5).

    Returns:
        numpy.ndarray: Khung hình sau khi vẽ.
    """
    h, w = frame.shape[:2]  # Lấy chiều cao và chiều rộng của khung hình

    # Duyệt qua các phát hiện (detections)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Độ tin cậy

        if confidence > confidence_threshold:
            value = int(detections[0, 0, i, 1])  # Nhãn đối tượng
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # Lấy tọa độ khung
            (startX, startY, endX, endY) = box.astype("int")

            # Đảm bảo tọa độ nằm trong giới hạn khung hình
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Màu sắc ngẫu nhiên
            color = COLORS[value % len(COLORS)]  # Lấy màu từ mảng COLORS
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Thêm nhãn và độ tin cậy lên khung hình
            label = f"{CLASSES[value]}: {confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 20  # Đảm bảo nhãn không bị cắt
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def drawCamera(camera, detections, confidence_threshold):
    """
    Vẽ kết quả phát hiện lên camera (luồng video trực tiếp).
        
    Args:
        camera (cv2.VideoCapture): Đối tượng camera (VideoCapture).
        detections (numpy.ndarray): Dữ liệu các phát hiện (có thể là kết quả từ mô hình).
        confidence_threshold (float): Ngưỡng tự tin để vẽ khung (mặc định: 0.5).
    """
    ret, frame = camera.read()

    if not ret:
        print("Không thể đọc frame từ camera.")
        return

    h, w = frame.shape[:2]  # Chiều cao và chiều rộng của frame

    # Duyệt qua các phát hiện (detections)
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Độ tin cậy

        if confidence > confidence_threshold:
            value = int(detections[0, 0, i, 1])  # Lấy class ID
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # Lấy tọa độ khung
            (startX, startY, endX, endY) = box.astype("int")

            # Đảm bảo tọa độ nằm trong giới hạn của frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Vẽ khung hình chữ nhật và nhãn lên frame
            color = COLORS[value % len(COLORS)]  # Lấy màu từ mảng COLORS
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Hiển thị nhãn và độ tin cậy
            label = f"{CLASSES[value]}: {confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 20  # Vị trí nhãn
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Hiển thị frame sau khi vẽ
    cv2.imshow("Camera Detection", frame)
