#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import re
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN, GRU, LSTM, Dense, Input, Embedding, Dropout, Activation, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers


# In[3]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
embedding_file = 'glove.6B.300d.txt'


# In[4]:


train_data.describe()


# In[5]:


test_data.describe()


# In[6]:


train_data.head()


# In[7]:


test_data.head()


# In[8]:


train_data.isnull().any()


# In[9]:


test_data.isnull().any()


# In[10]:


# Furhter actions on any columns is not required, because no columns has any missing data.


# In[11]:


classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_data[classes].values


# In[12]:


train_sentences = train_data["comment_text"].fillna("fillna").str.lower()
test_sentences = test_data["comment_text"].fillna("fillna").str.lower()


# In[13]:


max_features = 100000
max_len = 150
embed_size = 300


# In[14]:


tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(list(train_sentences))


# In[15]:


tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)


# In[16]:


train_sentences[1]


# In[17]:


tokenized_train_sentences[1]


# In[18]:


train_sentences[5]


# In[19]:


tokenized_train_sentences[5]


# In[20]:


train_padding = pad_sequences(tokenized_train_sentences, max_len)
test_padding = pad_sequences(tokenized_test_sentences, max_len)


# In[22]:


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[23]:


image_input = Input(shape=(max_len, ))
X = Embedding(max_features, embed_size, weights=[embedding_matrix])(image_input)
X = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(X)
# Dropout and R-Dropout sequence, inspired by Deep Learning with Python - Francois Chollet
avg_pl = GlobalAveragePooling1D()(X)
max_pl = GlobalMaxPooling1D()(X)
conc = concatenate([avg_pl, max_pl])
X = Dense(6, activation="sigmoid")(conc)
model = Model(inputs=image_input, outputs=X)


# In[24]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[25]:


saved_model = "weights_base.best.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]


# In[ ]:


batch_sz = 32
epoch = 2
model.fit(train_padding, y, batch_size=batch_sz, epochs=epoch, validation_split=0.1, callbacks=callbacks_list)


# In[25]:


test_values = model.predict([test_padding], batch_size=1024, verbose=1)


# In[26]:


sample_submission = pd.read_csv('sample_submission.csv')
sample_submission[classes] = test_values
sample_submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




