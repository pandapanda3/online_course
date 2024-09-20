# IMPORT LIBRARIES AND DATASETS

```python
!pip install --upgrade tensorflow-gpu==2.11
!pip install plotly
!pip install --upgrade nbformat
!pip install nltk
!pip install spacy # spaCy is an open-source software library for advanced natural language processing
!pip install WordCloud
!pip install gensim # Gensim is an open-source library for unsupervised topic modeling and natural language processing
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
# import keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to be monokai theme  
# this line of code is important to ensure that we are able to see the x and y axes clearly
# If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 

```

```python
# load the data
df_true = pd.read_csv("True.csv")
df_fake = pd.read_csv("Fake.csv")
df_true.isnull().sum()
df_fake.isnull().sum()
df_true.info()
df_fake.info()
```

# PERFORM EXPLORATORY DATA ANALYSIS

```python
# add a target class column to indicate whether the news is real or fake
df_true['isfake'] = 0
df_true.head()
df_fake['isfake'] = 1
df_fake.head()
```
```python
# Concatenate Real and Fake News
df = pd.concat([df_true, df_fake]).reset_index(drop = True)
df.drop(columns = ['date'], inplace = True)
# combine title and text together
df['original'] = df['title'] + ' ' + df['text']
df.head()
```

# PERFORM DATA CLEANING
```python
# download stopwords
nltk.download("stopwords")
# Obtain additional stopwords from nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
```

```python
# Remove stopwords and remove words with 2 or less characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result

# Apply the function to the dataframe
df['clean'] = df['original'].apply(preprocess)
```

```python
# Obtain the total words present in the dataset
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)

len(list_of_words)

# Obtain the total number of unique words
total_words = len(list(set(list_of_words)))

# join the words into a string
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))

```

# VISUALIZE CLEANED UP DATASET
```python
# plot the number of samples in 'subject'
plt.figure(figsize = (8, 8))
sns.countplot(y = "subject", data = df)

# Plot the count plot for fake vs. true news
plt.figure(figsize = (8, 8))
sns.countplot(y = "isfake", data = df)

```
```python
# plot the word cloud for text that is Real
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
```

```python
# plot the word cloud for text that is Fake
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
```

```python
# length of maximum document will be needed to create word embeddings 
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)
```

```python
# visualize the distribution of number of words in a text
import plotly.express as px
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
fig.show()
```

# PREPARE THE DATA BY PERFORMING TOKENIZATION AND PADDING

```python
# split data into test and train 
from sklearn.model_selection import train_test_split
from nltk import word_tokenize

x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

# Create a tokenizer to tokenize the words and create sequences of tokenized words
# Only the total_words words with the highest word frequency are reserved
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
# Converts text to a sequence of integers
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print(tokenizer.word_index)
```

```python
# Add padding can either be maxlen = 4406 or smaller number maxlen = 40 seems to work well based on results
# If a sequence is less than 40 in length, a 0 is added at the end of it.
padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post') 

for i,doc in enumerate(padded_train[:2]):
     print("The padded encoding for document",i+1," is : ",doc)
```

# BUILD AND TRAIN THE MODEL
```python
# Sequential Model
# The sequence model in Keras indicates that the layers of the model are stacked sequentially. The input of each layer comes from the output of the previous layer.
model = Sequential()

# embeddidng layer: The lexical index is mapped to 128-dimensional word vectors to capture semantic relationships between words.
model.add(Embedding(total_words, output_dim = 128))
# model.add(Embedding(total_words, output_dim = 240))


# Bi-Directional RNN and LSTM: Text sequences are processed through a bidirectional LSTM layer to capture contextual dependencies
# LSTM solves the gradient disappearance problem of common RNN by introducing gating mechanism (such as input gate, forget gate, output gate), and can effectively capture long distance dependent information.
model.add(Bidirectional(LSTM(128)))

# Dense layers: Further learning feature
model.add(Dense(128, activation = 'relu'))
# Binary prediction.
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
```

```python
y_train = np.asarray(y_train)
# train the model
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)

```

# ASSESS TRAINED MODEL PERFORMANCE
```python
# make prediction
pred = model.predict(padded_test)

# if the predicted value is >0.5 it is real else it is fake
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

# getting the accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)

# get the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize = (25, 25))
sns.heatmap(cm, annot = True)
```