#module
from utils.SoundPlayer import playSound
from models import DetectWay, DetectObject
#from servos import Servos
#from button import Button

import io
import socket
import time
import imutils
import numpy as np
import cv2


def save_image(image, value):
        global idxs 
        filename = dirr+"newdata/"+ direction +"/"+str(idxs[value])+".jpg"
        idxs[value]+=1
        cv2.imwrite(filename, image)       
idxs = [0, 0, 0]
dirr = "/home/haisgirl/Documents/mycode/Python/SmartGlassesForTheBlind/"
models_dirr = dirr + "models/"
sounds_dirr = dirr + "sounds/"
video_dirr = dirr + "video/"
MobileNetSSDDir = models_dirr+ "MobileNetSSD/"

weight_file = models_dirr + "model07012020.h5"
#weight_file = models_dirr + "model.hdf5"
video_output_file = video_dirr+"test.h264"
model = MobileNetSSDDir+"MobileNetSSD_deploy.caffemodel"
prototxt = MobileNetSSDDir +"MobileNetSSD_deploy.prototxt.txt"
directions=['center', 'left', 'right']
labels_name={'center':0,'left':1,'right':2}
# Model
DetectWay.initModel()
DetectWay.loadWeight(weight_file)
DetectWay.saveModel("/home/haisgirl/Documents/mycode/Python/SmartGlassesForTheBlind")
DetectObject.loadModel(prototxt, model)
camera = cv2.VideoCapture(0)
#camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # Cài đặt độ rộng
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Cài đặt độ cao

# Phát âm thanh để báo sẵn sàng
soundfile = sounds_dirr + 'ready.mp3'
playSound(soundfile)

 # Thiết lập codec và tạo đối tượng VideoWriter để lưu video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Chọn codec, 'XVID' thường dùng cho file .avi
# video_output_file = 'video_output.avi'  # Đường dẫn file video đầu ra
# out = cv2.VideoWriter(video_output_file, fourcc, 20.0, (640, 480))
print("Press s!!")
key = input()
while True:
        if key == 's':
            while True:
                print("s is pressed ")
                ret,frame = camera.read()
                if not ret:
                        print("Cannot open camera")
                        break
                rawCapture = cv2.resize(frame, (720,480))
                i = 0

                if (i % 10 == 0):
                    test_image = rawCapture
                    #Xu li anh
                    start_time1 = time.time()
                    test_image2 = test_image

                    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
                    test_image = cv2.resize(test_image,(128,128))

                    test_image = np.array(test_image)
                    test_image = test_image.astype('float32')
                    test_image /= 255

                    test_image = test_image.reshape(1,128,128,1)

                    # Du doan duong di
                    value = DetectWay.predict_class(test_image)
                    direction = directions[value]
                    print("This is: "+direction)
                    finish_time1 = time.time()
                    print("Total detectway-time: %f s", finish_time1 - start_time1)          
                    
                    # Phat qua tai nghe
                    soundfile = sounds_dirr + direction + '.mp3'
                    playSound(soundfile)
                    
                    # Xử lý nhận diện vật thể
                    start_time2 = time.time()
                    frame_resized = imutils.resize(test_image2, width=300)
                    detect = DetectObject.process(frame_resized)
                    #cho ket qua du
                    confidence = 0.8
                    DetectObject.predict(detect, confidence)
                    DetectObject.drawOnImage(frame_resized, detect, confidence)
                    DetectObject.drawOnImage(frame, detect, confidence)
                    # Hien thi text tren video
                    cv2.putText(frame, direction + " " + str(DetectWay.predict(test_image)[value]),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    finish_time2 = time.time()
                    print("Total detectobj-time: %f s", finish_time2 - start_time2)
                    # Hiển thị kết quả lên màn hình
                    cv2.imshow("File", frame)
                    # out.write(frame_resized)
                    #Lưu ảnh
                    save_image(frame, value)

                    process_time = finish_time2 - finish_time1
                    i+= int(process_time * 30)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    i += 1
                    key = cv2.waitKey(1) & 0xFF
                    if key == 'Q':
                        #Phat nhac qua tai nghe
                        soundfile = sounds_dirr + 'quit.mp3'
                        playSound(soundfile)

                        #dung camera
                        camera.release()
                        #out.release()
                        cv2.destroyAllWindows()
                        time.sleep(4)
                        exit()

