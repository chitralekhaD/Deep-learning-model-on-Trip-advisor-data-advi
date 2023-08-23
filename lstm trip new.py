#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np 
import pandas as pd
import tensorflow as tf 
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense , LSTM ,Embedding
from tensorflow.keras.optimizers import Adam
from keras.layers import Bidirectional


# In[28]:


trip = pd.read_csv(r"C:\Users\HP\Downloads\Trip_advisor_review.csv")


# In[29]:


trip.head()


# In[30]:


trip.isnull().sum()


# In[31]:


trip.shape


# In[61]:



from sklearn.model_selection import train_test_split


# In[62]:




trip_train , trip_test = train_test_split(trip , test_size = 0.2)


# In[63]:


trip_train_x = trip_train.iloc[: , 0]
trip_train_y= trip_train.iloc[: , 1]

trip_test_x = trip_test.iloc[: , 0]
trip_test_y= trip_test.iloc[: , 1]


# In[64]:


trip_train_x.shape


# In[65]:


trip_train_y


# In[66]:


trip_train_y= to_categorical(trip_train_y)


# In[67]:


trip_train_y.shape


# In[68]:


trip_train_y


# In[69]:


max_num_words = 11000


seq_len = 50


embedding_size = 100


# In[70]:



from keras.preprocessing.text                import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[71]:


tokenizer = Tokenizer(num_words = max_num_words)


# In[72]:


tokenizer.fit_on_texts(trip.Review)
trip_train_x = tokenizer.texts_to_sequences(trip_train_x)
trip_train_x = pad_sequences(trip_train_x , maxlen = seq_len)


# In[73]:


tokenizer.fit_on_texts(trip.Review)
trip_test_x = tokenizer.texts_to_sequences(trip_test_x)
trip_test_x = pad_sequences(trip_test_x , maxlen = seq_len)


# In[74]:


trip_train_x [8]


# In[75]:


trip_train_x.shape


# In[76]:


trip_test_x [10]


# In[77]:


trip_test_x.shape


# In[78]:


model = Sequential()
model.add(Embedding(input_dim = max_num_words,
                   input_length = seq_len,
                   output_dim = embedding_size))


# In[79]:


model.add(Bidirectional(LSTM(12)))
model.add(Dense(4 , activation = 'softmax'))

adam= Adam(lr = 0.003)
model.compile(optimizer = 'adam' , loss ='categorical_crossentropy' , metrics =['accuracy'])


# In[80]:



model.fit(trip_train_x , trip_train_y , epochs =8 , validation_split=0.2)


# In[81]:


model.summary()


# In[82]:


pred_values = model.predict(trip_test_x)


# In[83]:


pred_values


# In[84]:


pred_classes = np.argmax(pred_values ,axis =1)


# In[85]:


pred_classes


# In[86]:



from sklearn.metrics import confusion_matrix


# In[87]:


confusion_matrix(trip_test_y, pred_classes)


# In[ ]:





# In[88]:


accuracy_score(trip_test_y , pred_classes)*100


# In[60]:


#trip.Rating.replace({2:1 , 3:2 , 4:3, 5:3} , inplace=True)


# In[ ]:





# In[ ]:




