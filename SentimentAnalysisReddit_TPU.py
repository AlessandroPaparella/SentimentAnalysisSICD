# -*- coding: utf-8 -*-
"""redditTPU.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10wnMElpZxiV_RDwZbJVEiI4WbdyQGziD
"""
#pip install mega.py
#pip install transformers

import tensorflow as tf
import pandas as pd

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import os
from mega import Mega
from os.path import exists


def create_model():
    bert = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    inputs = tf.keras.layers.Input((None,), dtype=tf.int32)
    mask = tf.keras.layers.Input((None,), dtype=tf.int32)
    preds = bert(
        inputs,
        attention_mask=mask,
        training=True
    )[0]

    return tf.keras.Model([inputs, mask], preds)

model = create_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])


if(not exists("model.h5")):
    mega = Mega()
    m = mega.login()
    try:
        m.download_url('https://mega.nz/file/0aZUxbwD#FQ8RlKtBrtZ8k-EZYAy-CFBWzvszeCbzPwQh3OkrYIs')
    except:
        pass

model.load_weights("model.h5")

#pip install praw
 
import praw
import pandas as pd
 
reddit = praw.Reddit(
    client_id="QSmOVScqR5jgFjis7ps7mw",
    client_secret="v8up9w5kl4HJFWwDCjsOdn5x5Uyghw",
    password="Lsk3DnRATYWX99-",
    user_agent="testscript by u/franz1020",
    username="franz1020",
)

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

while True:
  choice = input("Inserisci:\n1 Per esaminare i commenti di un post\n2 Per esaminare i top post di un subreddit:\n") 

  if choice == '1':
    # PRENDO UN POST E VEDO I COMMENTI
    URL = input("Inserisci url del post:\n") 
    try:
      submission = reddit.submission(url=URL)
      comments = []
      submission.comments.replace_more(limit=0)
    
      for c in submission.comments:
          comments.append([c.body])

    except Exception:
        pass

    comments = pd.DataFrame(comments,columns=['data'])
    break

  elif choice == '2':
    # prendo i post di un subreddit
    chosen_subreddit = input("Inserisci il nome del subreddit:\n") 

    posts = []
    subreddit = reddit.subreddit(chosen_subreddit)

    for post in subreddit.hot(limit=200):
        posts.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
    posts = pd.DataFrame(posts,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])

    # PRENDO I TOP POST
    comments = []
    for i in range(posts.size):
      # Serve nel caso di URL non validi i quali vanno SKIPPATI
      try:
        comments.append([posts.get('body')[i]])
      except Exception: 
        pass 

    comments = pd.DataFrame(comments,columns=['data'])
    break

  else:
    print("Scelta non corretta")

import re
import string 

def cleanText(text):
  text = re.sub(r'@[A-Za-z0-9]+', '', text)
  text = re.sub(r'#', '', text)
  text = re.sub(r'[@[A-Za-z0-9]+]', '', text)
  text = re.sub(r'\s\(\s', '', text)
  text = re.sub(r'\s\($', '', text)
  text = re.sub(r'https?:\/\/\S+', '', text)
  return text

comments['data'] = comments['data'].apply(cleanText)

for i in range(len(comments)):
  if "[deleted]" in comments['data'][i] or "[removed]" in comments['data'][i] or '?' in comments['data'][i] or len(comments['data'][i]) < 3:
    comments = comments.drop(i)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

import re

def formatString(sentence):
  sentence = re.sub("(.{60})", "\\1\n", sentence, 0, re.DOTALL)
  return sentence

pred_sentences = comments["data"].values.tolist()

input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=128,pad_to_max_length=True) for sent in pred_sentences]
attention_mask = [[float(i>0) for i in seq] for seq in input_ids]
input_ids = tf.convert_to_tensor(input_ids)
attention_mask = tf.convert_to_tensor(attention_mask)
tf_outputs = model([input_ids, attention_mask])
tf_predictions = tf.nn.softmax(tf_outputs, axis=-1)
labels = ['Negative', 'Positive']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()

pred_labels = []
for i in range(len(pred_sentences)):
  pred_labels.append(labels[label[i]])
  pred_sentences[i] = formatString(pred_sentences[i])


data = {'Comment':pred_sentences,
        'Label':pred_labels}
        
df = pd.DataFrame(data)
from tabulate import tabulate
print(tabulate(df, headers = 'keys', tablefmt = 'fancy_grid'))

import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(pred_labels)
plt.show()
