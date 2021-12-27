#!/usr/bin/env python
# coding: utf-8

# # 3D Printer Capstone 
# In this project, the goal is to train a model to be able to recoginize failures in 3D printed layers.  To accomplish this, the model is fed training photos of good and bad layers of a 3D print then tested with different batches of photos to evaluate it's accuracy. A CNN model is chosen for this project as it appeared in related research papers, and gave the best results. An simpler method was also tried out using 1D arrays for each photo, the mean and standard of deviation of the array are compared against each other to find if it was capable of predicting failure without using image processing. However, this proves to be very inaccurate and was thus scrapped.

# 
# 

# ## Importing the Libraries

# In[1]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from numpy import asarray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Data Preprocessing

# In[2]:



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('Training_Top',
                                                 target_size = (256, 256),
                                                 batch_size = 36,
                                                 class_mode = 'categorical')

#Generate new images from exisiting images via the means of shearing zooming and fliping




# In[3]:


# for _ in range(5):
#     img, label = training_set.next()
#     print(img.shape)   #  (1,256,256,3)
#     plt.imshow(img[0])
#     plt.show()


# ### Preprocessing the Training Set

# ### Preprocessing the Test Set

# In[4]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Testing_Top',
                                            target_size = (256, 256),
                                            batch_size = 36,
                                            class_mode = 'categorical')


# ## Building the CNN Model
# Build CNN model with 8 layers with 2 convolutional layers. 

# ### Step 1 - Initialising the Model

# In[5]:


Model = tf.keras.models.Sequential()


# ### Step 2 - Adding First Convolution Layer 

# In[6]:


Model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=6, activation='relu', input_shape=[256, 256, 3]))


# ### Step 3 - Pooling the First Layer

# In[7]:


Model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Step 4 - Adding a Second Convolutional Layer

# In[8]:


Model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))


# ### Step 5 - Pooling the Second Layer

# In[9]:


Model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# ### Step 6 - Flattening

# In[10]:


Model.add(tf.keras.layers.Flatten())


# ### Step 7 - Full Connection

# In[11]:


Model.add(tf.keras.layers.Dense(units=128, activation='relu'))


# ### Step 8 - Output Layer

# In[12]:


Model.add(tf.keras.layers.Dense(units=4, activation='sigmoid'))

Model


# In[ ]:





# ### Step 9 - Compiling the CNN

# In[13]:


Model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])


# ## Training the CNN and Evaluation

# In[14]:


Model.fit(x = training_set, validation_data = test_set, epochs = 10)


# In[15]:



## Making a Prediction


# In[24]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('G_code.jpg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = Model.predict(test_image)
training_set.class_indices
print("The probobility of No defect is:",result[0][0]*100,'%')
print("The probobility of Misalignment is:",result[0][2]*100,'%')
print("The probobility of Inconsistance is:",result[0][3]*100,'%')


# In[ ]:





# In[ ]:




