import os
import csv
import cv2
import numpy as np
from keras.layers import Input, Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Dropout
from keras.models import Sequential
from PIL import Image
import io
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import boto3
import sys, getopt
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
images = []
labels = []
correction = 0.2

def load_data_from_s3(path):
    data =[]

    with open('{}/driving_log.csv'.format(path)) as csvfile:
        reader = csv.reader(csvfile)
        line_count = 0
        for line in reader:
            if line_count>0:
                steering_angle = float(line[3])
                center = [line[0], steering_angle]
                left= [line[1], steering_angle+correction]
                right= [line[2], steering_angle-correction]
                flip = [line[0], -1*steering_angle, 'flip']
                data.extend((center,left,right, flip))
            line_count+=1
    return data


def get_image(bucket,sample):
    try:
        image_name = sample[0].split('/')[-1]
        key= 'IMG/{}'.format(image_name)
        image_obj = s3.Object(bucket, key)
        image = io.BytesIO(image_obj.get()['Body'].read())
        image = Image.open(image)
        image= np.asarray( image, dtype="int32" )
        if sample[-1] == 'flip':
            image = np.fliplr(image)
        return image
    except Exception as err:
        print(err)


def generator(bucket,samples, batch_size = 32):
    while True:
        for offset in range(0,len(samples), batch_size):
            start = offset
            end = offset+batch_size
            batch_sample = samples[start:end]
            images=[]
            angles=[]
            for sample in batch_sample:
                #get images
                image = get_image(bucket,sample)

                images.append(image)
                #get steering angles
                steer = float(sample[1])
                angles.append(steer)

            x = np.array(images)
            y = np.array(angles)
            yield shuffle(x,y)

#model architecture
def run_model(train_data, validation_data, bucket):
    train_generator = generator(bucket,train_data, batch_size = 32)
    validation_generator = generator(bucket,validation_data, batch_size =32)
    model = Sequential()
    model.add(Lambda(lambda train_generator:(train_generator/250.0)-0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator,samples_per_epoch=len(train_data),nb_epoch=5,nb_val_samples=len(validation_data), validation_data = validation_generator, verbose=1)
    return model, history_object


def main(argv):
    bucket = ''

    try:
        opts, args = getopt.getopt(argv,"hb:",["bucket="])
    except getopt.GetoptError:
        print('model.py -b <bucketname>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('model.py -b <bucketname>')
            sys.exit()
        elif opt in ("-b", "--bucketname"):
            bucket = arg

        if bucket:
            try:
                print("\u2713 Bucket recieved: {}".format(bucket))
                print(" Loading Data ... ")
	        # create a directory to store driving_log.csv of the target bucke
                if not os.path.exists(bucket):
                    os.makedirs(bucket)
                    s3_client.download_file(bucket, 'driving_log.csv', '{}/driving_log.csv'.format(bucket))
                    data = load_data_from_s3(bucket)
                    train_data, validation_data = train_test_split(data, test_size=0.2)

                    print(" ----------------------------")
                    print("| Data : {} \u2713".format(len(data)))
                    print("| Training Data (80%): {} \u2713".format(len(train_data)))
                    print("| Validation Data (20%): {} \u2713".format(len(validation_data)))
                    print(" ----------------------------")

                    print(" Training the model started ... ")
                    model, history_object= run_model(train_data, validation_data, bucket)
                    model.save('{}/model.h5'.format(bucket))

                    print('Generating loss plot...')
                    plt.plot(history_object.history['loss'])
                    plt.plot(history_object.history['val_loss'])
                    plt.title('model mean squared error loss')
                    plt.ylabel('mean squared error loss')
                    plt.xlabel('epoch')
                    plt.legend(['training set', 'validation set'], loc='upper right')
			
                    plt.savefig('{}/loss.png'.format(bucket))

            except Exception as err:
                print(err)
        else:
            print('please pass a bucket to model.py "model.py -b <bucketname>"')

if __name__ == "__main__":
    main(sys.argv[1:])


