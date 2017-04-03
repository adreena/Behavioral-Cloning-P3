import csv
import os
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras.models import Sequential
images = []
labels = []

correction = 0.2
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        image_name = line[0].split('/')[-1]
        correct_path = "./data/IMG/"+image_name
        image = cv2.imread(correct_path)
        images.append(image)
  
        image_left_name = line[1].split('/')[-1]
        image_right_name = line[2].split('/')[-1]
        image_left = cv2.imread("./data/IMG/{}".format(image_left_name))
        image_right = cv2.imread("./data/IMG/{}".format(image_right_name))
        images.append(image_left)
        images.append(image_right)

        images.append(cv2.flip(image,1))

        steer = float(line[3])
        labels.append(steer)
        labels.append(steer+correction)
        labels.append(steer-correction)
        labels.append(steer*-1.0)

x = np.array(images)
y = np.array(labels)

print(x.shape)
print(y.shape)
model = Sequential()
model.add(Lambda(lambda x:(x/250.0)-0.5, input_shape=(x[0].shape)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
if os.path.isfile('features.h5'):
    print('loading features')
    model.load_weights('features.h5')
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, validation_split=0.2, shuffle= True, nb_epoch=15)
model.save_weights('features.h5')
model.save('model.h5')

print('Done')
