from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import os

from google.colab import drive
drive.mount('/content/drive')

import tarfile 
file = tarfile.open('drive/MyDrive/140.tar.gz') 
  
# extracting file 
file.extractall('./') 
  
file.close() 

# The shutil module offers a number of high-level 
# operations on files and collections of files.
import shutil
# Create main directory path ("/140")
dataset="./140"
main_dir = os.path.join(os.path.dirname(dataset), '140')
# Create sub directory path ("/140/train")
train_dir = os.path.join(main_dir, 'train')
# View the final train folder
print(os.listdir(train_dir))

# We create a training dataset and a validation 
# dataset from our "140/train" directory with a 80/20 split.
train = tf.keras.preprocessing.text_dataset_from_directory(
    '140/train', batch_size=1119999, validation_split=0.2, 
    subset='training', seed=123)
val = tf.keras.preprocessing.text_dataset_from_directory(
    '140/train', batch_size=223999, validation_split=0.2, 
    subset='validation', seed=123)
#create test set
test = tf.keras.preprocessing.text_dataset_from_directory(
    '140/test', batch_size=480000, seed=123)

import pandas as pd
for i in train.take(1):
  train_feat = i[0].numpy()
  train_lab = i[1].numpy()

train = pd.DataFrame([train_feat, train_lab]).T
train.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
train['DATA_COLUMN'] = train['DATA_COLUMN'].str.decode("utf-8")
train.head()

for j in val.take(1):
  val_feat = j[0].numpy()
  val_lab = j[1].numpy()

val = pd.DataFrame([val_feat, val_lab]).T
val.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
val['DATA_COLUMN'] = val['DATA_COLUMN'].str.decode("utf-8")
val.head()

for t in test.take(1):
  test_feat = t[0].numpy()
  test_lab = t[1].numpy()

test = pd.DataFrame([test_feat, test_lab]).T
test.columns = ['DATA_COLUMN', 'LABEL_COLUMN']
test['DATA_COLUMN'] = test['DATA_COLUMN'].str.decode("utf-8")
test.head()

#cleaning data
import re
import string 

def cleanText(text):
  text = re.sub(r'@[A-Za-z0-9]+', '', text)
  text = re.sub(r'#', '', text)
  text = re.sub(r'https?:\/\/\S+', '', text)
  return text

train['DATA_COLUMN'] = train['DATA_COLUMN'].apply(cleanText)
val['DATA_COLUMN'] = val['DATA_COLUMN'].apply(cleanText)
test['DATA_COLUMN'] = test['DATA_COLUMN'].apply(cleanText)

#download tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#configure TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver) 

#tokenize string and create tensors
MAX_LEN = 128

train_input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in train["DATA_COLUMN"]]
## Create attention mask
train_attention_mask = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
train_attention_mask = [[float(i>0) for i in seq] for seq in train_input_ids]
y_train = train["LABEL_COLUMN"]

val_input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in val["DATA_COLUMN"]]
## Create attention mask
val_attention_mask = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
val_attention_mask = [[float(i>0) for i in seq] for seq in val_input_ids]
y_val = val["LABEL_COLUMN"]

#shuffle train and validation set
import random

def shuffle(input_ids, attention_mask, y):
  dt = zip(input_ids, attention_mask, y)
  dt= list(dt)
  random.shuffle(dt)
  l = list(zip(*dt))
  return l[0], l[1], l[2]

train_input_ids, train_attention_mask, y_train = shuffle(train_input_ids, train_attention_mask, y_train)
val_input_ids, val_attention_mask, y_val = shuffle(val_input_ids, val_attention_mask, y_val)

def createTensors(input_ids, attention_mask, y):
  return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask), tf.convert_to_tensor(y, dtype=tf.int64)

train_input_ids, train_attention_mask, y_train = createTensors(train_input_ids, train_attention_mask, y_train)
val_input_ids, val_attention_mask, y_val = createTensors(val_input_ids, val_attention_mask, y_val)


#TRAINING

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

with strategy.scope():
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
    
model.fit(
    [train_input_ids, train_attention_mask],
    y_train,
    validation_data = ([val_input_ids, val_attention_mask], y_val),
    validation_steps=val_input_ids.shape[0] // 16,
    batch_size=16,
    epochs=2
)


#save model weights
model.save_weights("model.h5")
shutil.copy("model.h5", "drive/MyDrive/model.h5")


#EVALUATION

test_input_ids = [tokenizer.encode(sent, add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in test["DATA_COLUMN"]]

## Create attention mask
test_attention_mask = []
## Create a mask of 1 for all input tokens and 0 for all padding tokens
test_attention_mask = [[float(i>0) for i in seq] for seq in test_input_ids]
y_test = test["LABEL_COLUMN"]

test_input_ids, test_attention_mask, y_test = shuffle(test_input_ids, test_attention_mask, y_test)
test_input_ids, test_attention_mask, y_test = createTensors(test_input_ids, test_attention_mask, y_test)

model.evaluate([test_input_ids, test_attention_mask], y_test, steps=test_input_ids.shape[0] // 16)
