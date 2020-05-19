#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (224, 224))
       # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory with unique name
        file_name_path = 'C://Users//Rajesh Bhalla//Desktop//facerecog1//nitish + str(count)' + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")


# In[2]:


from keras.applications import MobileNet

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[3]:


def lw(bottom_model, num_classes):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model


# In[4]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 3 (Young, Middle, Old)
num_classes = 10

FC_Head = lw(MobileNet, num_classes)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())


# In[5]:


from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'C://Users//Rajesh Bhalla//Desktop//facerecog1//training//'
validation_data_dir = 'C://Users//Rajesh Bhalla//Desktop//facerecog1//valid//'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 
# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 32
 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


# In[6]:


from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

                     
checkpoint = ModelCheckpoint("face_recog1.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# We use a very small learning rate 
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

# Enter the number of training and validation samples here
nb_train_samples = 227
nb_validation_samples = 81

# We only train 5 EPOCHS 
epochs = 25
batch_size = 64

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)


# In[7]:


from keras.models import load_model

classifier = load_model('face_recog1.h5')


# In[ ]:


import os

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_dict = {"[0]": "Nitish",
            "[1]":"Stranger"}

face_dict_n = {"nitish":"Nitish",
            "stranger":  "Stranger"}
def draw_test(name, pred, im):
    face=face_dict[str(pred)]
    BLACK = [0,0,0]
    print(im.shape[0])
    expanded_image = cv2.copyMakeBorder(im, 0, 0, 0, input_im.shape[0] ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, str(pred), (52, 70) , cv2.FONT_HERSHEY_COMPLEX_SMALL,4, (0,255,0), 2)
    cv2.imshow(name, expanded_image)

def getRandomImage():
    path = 'C://Users//Rajesh Bhalla//Desktop//facerecog1//valid//'
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + str(path_class))
    file_path = path + path_class

    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    print(file_path+"/"+image_name)
    return cv2.imread(file_path+"/"+image_name)    

for i in range(0,10):
    input_im = getRandomImage()
    input_original = input_im.copy()
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    cv2.imshow("Test Image", input_im)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    ## Get Prediction
    print(classifier.predict(input_im, 1, verbose = 0))
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)

    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

    
cv2.destroyAllWindows()


# In[ ]:


import os

os.walk("C://Users//Rajesh Bhalla//Desktop//facerecog1//valid//")


# In[ ]:


import os
path = 'C://Users//Rajesh Bhalla//Desktop//facerecog1//valid//'
folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
print(folders)

random_directory = np.random.randint(0,len(folders))
print(folders[random_directory])


# In[ ]:



import os

import cv2

import numpy as np

def getRandomImage():
    path = ''
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    print("Class - " + str(random_directory))
    path_class = folders[random_directory]
    file_path = path + path_class

    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    print(file_path+"/"+image_name)
    return cv2.imread(file_path+"/"+image_name)

image = getRandomImage()
cv2.imshow("test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




