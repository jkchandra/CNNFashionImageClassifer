import numpy as np
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
#from __future__ import print_function
import keras
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, ZeroPadding2D, Convolution2D
from keras.utils import to_categorical
from skimage import io, transform
import glob
import os
import tensorflow as tf
import time
from PIL import Image
import re
import csv
from keras.preprocessing.image import ImageDataGenerator

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def importTrain2():

    # trainset path
    path = 'INSERT PATH'
    # resize all pictures to 128 * 128, subjective to adjustment
    w = 299
    h = 299
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    paths = []
    count = 0
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            paths.append(im)
        paths.sort(key=natural_keys)

        for im in paths:
            img = cv2.imread(im)
            if (img is None):
                print('The image:%s cannot be identified.' % im )
                continue
            img1 = cv2.resize(img, (w, h))
            imgs.append(img1)
            labels.append(idx)
            count += 1
        print("Finished importing folder %s" % folder)
        paths = []
    imgsnp = np.asarray(imgs)
    labelsnp = np.asarray(labels)
    result = [imgsnp, labelsnp]
    return result


def importTest2():
    # testset path
    path = test_path
    # resize all pictures to 128 * 128, subjective to adjustment
    w = img_width
    h = img_height
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    paths = []
    count = 0
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            paths.append(im)
        paths.sort(key=natural_keys)

        for im in paths:
            img = cv2.imread(im)
            if (img is None):
                print('The image:%s cannot be identified.' % im )
                continue
            img1 = cv2.resize(img, (w, h))
            imgs.append(img1)
            labels.append(idx)
            count += 1
        print("Finished importing folder %s" % folder)
        paths = []
    imgsnp = np.asarray(imgs)
    labelsnp = np.asarray(labels)
    result = [imgsnp, labelsnp]
    return result

# Used to create a CNN model
def createModel1():

    # Following the VCG16 Architecture for CNN

    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(3,128,128)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def createModel2():
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (7, 7), padding='same', activation='relu'))
    model.add(Conv2D(64, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (7, 7), padding='same', activation='relu'))
    model.add(Conv2D(64, (7, 7), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model

def exportCSV(pred):
    with open('predict_result.csv', 'w', newline='') as f:
        header = ['id','category']
        # input headers name.
        writer = csv.DictWriter(f, header)
        writer.writeheader()

        for i in range(0,len(pred)):
            writer.writerow({
                'id': i+1,
                'category':pred[i]
            })

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend('train', loc='upper left')
    plt.show()
    plt.savefig('Accuracy.png')

# summarize history for loss
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend('train', loc='upper left')
    plt.show()
    plt.savefig('Loss.png')

#Eliminate corrupted files and counts the number of images for the train set
def openTrain(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    count = 0
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = cv2.imread(im)
            if (img is None):
                print('The image:%s cannot be identified.' % im)
                newname = im.replace('.jpg', 'txt')  # convert file type, so it won't affect fit_generator
                os.rename(im, newname)
                continue
            count += 1
        print("Finished reading folder %s" % folder)
    print("Total number of uncorrupted images: %d" % count)
    return count

#Builds the Datagen, DataGenerator and the Class Dictionary
def build_gen(source):
    datagen = ImageDataGenerator(rescale = 1./255, zoom_range=0.3, rotation_range=30, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True, vertical_flip=False)
    generator = datagen.flow_from_directory(
        source,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')
    class_dictionary = generator.class_indices
    return generator, class_dictionary

# MAIN APP STARTS HERE --------------------------------------------------
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
#session = tf.Session(config=config)
set_session(tf.Session(config=config))

np.random.seed(1337)
#train = importTrain2()
train_path = 'INPUT PATH HERE'
test_path = 'INPUT PATH HERE'
img_height = 244
img_width = 244
NB_IV3_LAYERS_TO_FREEZE = 172

countTrain = openTrain(train_path)
test = importTest2()

#trainImages = train[0]
#trainLabels = train[1]
testImages = test[0]
testLabels = test[1]

#classes = np.unique(trainLabels)
#nClasses = len(classes)

#print(trainImages.shape)
print(testImages.shape)
#nRows, nCols, nDims = trainImages.shape[1:]

#trainData = trainImages.reshape(trainImages.shape[0], nRows, nCols, nDims)
testData = testImages.reshape(testImages.shape[0], img_width, img_height, 3)
input_shape = (img_width, img_height, 3)

# Change to float datatype
#trainData = trainData.astype('float32')
testData = testData.astype('float32')

# Scale the data to lie between 0 to 1
#trainData /= 255
testData /= 255

# Change the labels from integer to categorical data
#trainLabels_onehot = to_categorical(trainLabels)
testLabels_onehot = to_categorical(testLabels)

#model1 = createModel2()

#Creating model using VCG19 -----------------------------------------------------------------------------------------------------------------------------
#model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',  input_shape=)
model = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_shape=(img_width,img_height,3))
#model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = )
#model1 = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=(128,128,3), pooling=None, classes=nClasses)

#hyperparameters
batch_size = 256
epochs = 50
sPerEpoch = int(float(countTrain)/float(batch_size))

#Freezing layer
for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
    layer.trainable = False
for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
    layer.trainable = True

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(18, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

#compiling the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

#model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_final.summary()

#Data Generator ----------------------------------------------------------------------------------------------------------------------------------------------

#Builds the generator
generator, class_dictionary  = build_gen(train_path)
# Trains the model
#model1.fit(trainData, trainLabels_onehot, batch_size=batch_size, epochs=epochs, verbose=1)
history = model_final.fit_generator(generator = generator,
                              steps_per_epoch = sPerEpoch,
                              epochs = epochs)



#Predicts the output of the model
prob = model_final.predict(testData)
pred = prob.argmax(axis=-1)
print(pred)
exportCSV(pred)

#Saves the model
model_final.save('my_model.h5')

#Plots the accuracy and the loss function of the model
#plots the graphs
plot_loss(history)
plot_acc(history)


#model2.evaluate(test_data, test_labels_one_hot)
