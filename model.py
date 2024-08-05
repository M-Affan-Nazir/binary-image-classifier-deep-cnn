import tensorflow as tf 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt


def imageAugmentation():
    #to avoid overfitting; we will apply transformation on the images (zoom, stretch, rotation, flips etc). This is called image augmentation
    AugmentationGeneratorForTraining = ImageDataGenerator( rescale=1./255, #dividing the value of each pixel by 255
                                                shear_range=0.2,
                                                zoom_range=0.2,
                                                horizontal_flip=True)
    
    trainingDataset = AugmentationGeneratorForTraining.flow_from_directory("./dataset/training/",
                                                                target_size = (64,64),   #all images will be resized to 64 by 64 pixels.
                                                                batch_size=32,
                                                                class_mode="binary")  #output is either cancer or non-cancer; so binary (1,0) seems good
    
    AugmentationGeneratorForTesting = ImageDataGenerator(rescale=1./255)
    testingDataset = AugmentationGeneratorForTesting.flow_from_directory("./dataset/test",
                                                                         target_size = (64,64),
                                                                         batch_size=32,
                                                                         class_mode="binary")
    
    modelArchetecture(trainingDataset,testingDataset)

def modelArchetecture(trainingDataset, testingDataset):
    cnn = tf.keras.models.Sequential()

    #Adding 2 Convolution and Pooling Layers:
    cnn.add( tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=[64,64,3]) ) #taking the maximum value of pixel wala concept / pooling.  Number of filters = 32, the filter matrix size = 3x3, input_shape = input dimensions of the image! we input an image scaled down to 64 by 64! Since the pictures are colored (RGB); the third value is 3; indicating RGB! If image was black and white; this would be 1.
    cnn.add( tf.keras.layers.MaxPool2D(pool_size=2, strides = 2) ) #pool_size = size of the overlay matrix, strides, how many pixels you want the pooling matrix! If its a 2by2; you wanna move it my 2 pixels kyounkay if you move by 1; it will take a row/column again into consideration. If you take 3; it will 'miss one' column/row.
    cnn.add( tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
    cnn.add( tf.keras.layers.MaxPool2D(pool_size=2, strides = 2) ) 

    #Flattening:
    cnn.add( tf.keras.layers.Flatten() )  #Flattens the data being fed into it into a Nx1 vector! Since the images are [64,64,3]; the .Flatten will return a [64x64x3 =] 12288x1 vector for each image! (First 3 values in the vector will be the Red,Green,Blue value for the first pixel of the image)

    #Adding Neural Networks:
    cnn.add( tf.keras.layers.Dense(units=128, activation="relu") )
    cnn.add( tf.keras.layers.Dense(units=128, activation="relu") )
    cnn.add( tf.keras.layers.Dense(units=128, activation="relu") )

    #Output Layer:
    cnn.add( tf.keras.layers.Dense(units=1, activation="sigmoid") )

    #Compile
    cnn.compile(optimizer="adam", loss="binary_crossentropy", metrics= ['accuracy'])

    modelTraining(cnn,trainingDataset,testingDataset)


def modelTraining(cnnModel, trainingDataset, testingDataset):
    cnnModel.fit(x=trainingDataset, validation_data=testingDataset, epochs=27)
    #cnnModel.save("./trainedModel")

