#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



import nltk
nltk.download('stopwords')


# In[42]:


# printing the stopwords in English
print(stopwords.words('english'))


# In[43]:


news_dataset=pd.read_csv('train.csv')
news_dataset.head()


# In[44]:


news_dataset.shape


# In[45]:


news_dataset.isnull().sum()


# In[46]:


news_dataset=news_dataset.fillna('')


# In[47]:


news_dataset['content']=news_dataset['author']+' '+news_dataset['title']


# In[48]:


news_dataset['content']


# In[49]:


X=news_dataset.drop(columns='label',axis=1)
y=news_dataset.label


# In[50]:


X


# In[51]:


y


# In[52]:


port_stem=PorterStemmer()


# In[53]:


def stemming(content):
    stemmed_content=re.sub('[^a-zA-Z]',' ',content)
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content


# In[54]:


news_dataset['content']=news_dataset['content'].apply(stemming)


# In[55]:


print(news_dataset['content'])


# In[56]:


X=news_dataset['content'].values
y=news_dataset['label'].values


# In[57]:


print(X,y,y.shape)


# In[58]:


vectorizer=TfidfVectorizer()
vectorizer.fit(X)

X=vectorizer.transform(X)


# In[59]:


print(X)


# In[60]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[61]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[72]:


y_pred_train=model.predict(X_train)
model_train_accuracy=accuracy_score(y_pred_train,y_train)


# In[73]:


print(model_train_accuracy)


# In[74]:


y_pred_test=model.predict(X_test)
model_test_accuracy=accuracy_score(y_pred_test,y_test)


# In[75]:


print(model_test_accuracy)


# In[78]:


print(y_test[3])


# In[82]:


X_new = X_test[3]
prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[ ]:




