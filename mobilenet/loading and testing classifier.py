#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#####Loading classifier
from keras.models import load_model

classifier = load_model('face_recog1.h5')


# In[ ]:


##Testing our classifier on test images
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
    path = ''
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

os.walk("")


# In[ ]:


import os
path = ''
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

