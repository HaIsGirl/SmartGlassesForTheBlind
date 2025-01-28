from tf_keras.models import load_model
import numpy as np
from tf_keras.models import Sequential 
from tf_keras.layers import Dense, Dropout, Activation, Flatten 
from tf_keras.layers import Conv2D, MaxPooling2D 
import tensorflow as tf
# Define the number of classes
model = None
num_classes = 3
labels_name={'center':0,'left':1,'right':2}
directions={'center', 'left', 'right'}


def initModel():
	global model
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128,128,1)))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5)) 
	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

def viewModelConfig():
	global model
	# Viewing model_configuration

	model.summary()
	model.get_config()
	model.layers[0].get_config()
	model.layers[0].input_shape         
	model.layers[0].output_shape            
	model.layers[0].get_weights()
	np.shape(model.layers[0].get_weights()[0])
	model.layers[0].trainable
	
# load model
def loadWeight(file_dir):	
	global model
	model.load_weights(file_dir)
	
def predict_class(image):
    global model
    # Dự đoán đầu ra với mô hình, trả về xác suất của tất cả các lớp
    predictions = model.predict(image)
    
    # Lấy chỉ số của lớp có xác suất cao nhất
    index_label = np.argmax(predictions, axis=1)[0]
    
    return index_label
	
def predict(image):
	global model
	acc = model.predict(image)[0];
	return acc

def saveModel(file_dir):
    global model
    model.save("/home/haisgirl/Documents/mycode/Python/SmartGlassesForTheBlind/mymodels/model.h5")

