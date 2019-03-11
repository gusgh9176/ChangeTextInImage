
# coding: utf-8

# In[1]:


from __future__ import print_function
import keras
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import cv2
import os
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

p_file_list = os.listdir("./positive")
y_data = []

for i in range(len(p_file_list)):
    y_data.append([0])
    
n_file_list = os.listdir("./negative")

for i in range(len(n_file_list)):
    y_data.append([1])

file_list = p_file_list + n_file_list
sfl_num = list(range(len(file_list)))
np.random.shuffle(sfl_num)

tmp_std = int(len(sfl_num)/10 * 7)
tr_sfl_num = sfl_num[:tmp_std]
ts_sfl_num = sfl_num[tmp_std:]


# In[2]:


x_train = []
y_train = []
x_test = []
y_test = []

for element in tr_sfl_num:
    if y_data[element] == [0]:
        tmp_str = str("./positive/") + file_list[element]
    else :
        tmp_str = str("./negative/") + file_list[element]
    x_train.append(cv2.imread(tmp_str, cv2.IMREAD_GRAYSCALE))
    y_train.append(y_data[element])
    
for element in ts_sfl_num:
    if y_data[element] == [0]:
        tmp_str = str("./positive/") + file_list[element]
    else :
        tmp_str = str("./negative/") + file_list[element]
    x_test.append(cv2.imread(tmp_str, cv2.IMREAD_GRAYSCALE))
    y_test.append(y_data[element])


# In[3]:


for i in range(len(x_train)):
    x_train[i] = cv2.resize(x_train[i], (200, 200))
    x_train[i] = x_train[i] / 255
    
print(x_train[1].shape)

for i in range(len(x_test)):
    x_test[i] = cv2.resize(x_test[i], (200, 200))
    x_test[i] - x_test[i]/255
    
x_train= np.asarray(x_train)
x_test = np.asarray(x_test)
    
x_train = x_train.reshape(len(y_train),200,200,1)
x_test = x_test.reshape(len(y_test),200,200,1)

y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)


# In[4]:


#create model
model = Sequential()
#add model layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,batch_size=16,epochs=20, verbose=1, validation_split=0.2, shuffle=True)


# In[ ]:


model_json = model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)


# In[ ]:


model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


model.evaluate( x_test, y_test)


# In[ ]:


#from keras.models import model_from_json

#json_file = open("model.json", "r")
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

#loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

