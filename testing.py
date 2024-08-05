import tensorflow as tf 
import numpy as np 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import matplotlib.pyplot as plt


def testing():
    cnnModel = tf.keras.models.load_model("./trainedModel")  #to see image; use pyplot.imshow(). Its a blurry image; cuz its 64x64 pixels only.
    print(cnnModel.summary())
    img = image.load_img("./dataset/single_prediction/cancer.jpg", target_size=(64,64))  #target size= resize image to 64x64 pixels
    imgArray = image.img_to_array(img) #turns the image to a numpy 3D array. 3rd dimension for RGB value.
    imgArrayWithBatchDimension = np.expand_dims(imgArray, axis = 0) # *** When we input training data to model; we data as batches. Our size was 32 batches; which means 32 images were being fed into the neural network simultanously! Therefore, the first dimension of the input array is batch. input_data[0] refers to the data of the first image and input_data[20] refers to the data of the 21st image; all the up to 32! The model; therefore expects an extra dimension of batches; therefore we expand the array. axis = 0 specifies that we want to add the extra dimension AT THE START; because ofcourse batch dimension if first! iskay agay images ka data ata na. Secondly Note; here we have only 1 image in the batch that we want to predict. If we wanted to predict about 2 images; this batch array would had 2 images then; thus the model would had returned 2 seperate prediction! [prediction would also be in a higher dimension array, like [ [0.7,0.3] , [0.1,0.9] ]. First index has probability for cat and dog for first image; second array has prob for cat and dog of 2nd image. The global array is the 'Batch' Array!
    
    prediction= cnnModel.predict(imgArrayWithBatchDimension)
    print(prediction)



os.system("cls")

testing()