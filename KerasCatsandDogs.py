#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#import PIL


# In[2]:


# Initialising the CNN
classifier = Sequential()
# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# second pooling layer
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[3]:


# Step 3
classifier.add(Flatten())

# Step 4 
classifier.add(Dense(units=128, activation = 'relu'))
classifier.add(Dense(units=1, activation = 'sigmoid'))


# In[4]:


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[5]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)


# In[6]:


training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')


# In[7]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[12]:


#TRAINING

classifier.fit(training_set,
                        #steps_per_epoch = 24,
                        epochs = 20,
                        validation_data = test_set,
                        #validation_steps = 50
              )


# In[24]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_9.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)


# In[ ]:




