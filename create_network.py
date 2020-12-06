import collections
import pathlib
import re
import string

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import utils
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import tensorflow_datasets as tfds
import tensorflow_text as tf_text

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

batch_size_choice = 100
seed_choice = 1312

raw_train_ds = preprocessing.text_dataset_from_directory(
    "train",
    batch_size=batch_size_choice,
    validation_split=0.2,
    subset='training',
    seed=seed_choice)

raw_val_ds = preprocessing.text_dataset_from_directory(
    "train",
    batch_size=batch_size_choice,
    validation_split=0.2,
    subset='validation',
    seed=seed_choice)

raw_test_ds = preprocessing.text_dataset_from_directory("test", batch_size=batch_size_choice)

#Note: to avoid indexing out of bounds, must provide enough examples to fill an entire batch.


#Sets the size of the vocabulary of our text data samples.
VOCAB_SIZE = 30

#Sets the maximum length of our text samples. Text samples which are shorter than this will be padded out to be this length.
MAX_SEQUENCE_LENGTH = 100

#This layer 'vectorizes' our text data into number data that the neural net can actually use.
int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    split='whitespace',
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)



#Here we call 'adapt' to turn our raw preprocessed dataset into int data which can be used by our neural net.
train_text = raw_train_ds.map(lambda text, labels: text)
print(int_vectorize_layer.get_vocabulary())

int_vectorize_layer.adapt(train_text)


def int_vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return int_vectorize_layer(text), label


text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]


#Here we map our raw datasets to useable int data with our int_vectorize_text function.
int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

AUTOTUNE = tf.data.experimental.AUTOTUNE

#int_train_ds = int_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#int_val_ds = int_val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#int_test_ds = int_test_ds.cache().prefetch(buffer_size=AUTOTUNE)



def create_model(vocab_size, num_labels):
  model = tf.keras.Sequential([
      int_vectorize_layer,
      
      
      #layers.Embedding(vocab_size, 128, mask_zero=True),
      #layers.Bidirectional(layers.LSTM(128,  return_sequences=True)),
      #layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
      #layers.Bidirectional(layers.LSTM(32)),
      #layers.Dense(128, activation='relu'),
      #layers.Dropout(0.5),
      
      layers.Embedding(vocab_size, 64, mask_zero=True),
      layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
      layers.Bidirectional(tf.keras.layers.LSTM(32)),
      layers.Dense(64, activation='relu'),
      layers.Dropout(0.5),

      #layers.Embedding(vocab_size, 64, mask_zero=True),
      #layers.LSTM(64),
      #layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
      #layers.GlobalMaxPooling1D(),

      #layers.Embedding(vocab_size, 128, mask_zero=True),
      #layers.Conv1D(128, 5, padding="valid", activation="relu", strides=2),
      #layers.GlobalMaxPooling1D(),
      #layers.LSTM(128),
      #layers.Dropout(0.5),
      
      #layers.Dense(32),
      #layers.Conv1D(32, 5, padding="valid", activation="relu", strides=2),
      #layers.LSTM(32),
      #layers.GlobalMaxPooling1D(),

      #layers.Embedding(512, 10),
      #layers.Dropout(0.2),
      #layers.GlobalAveragePooling1D(),
      #layers.Dropout(0.2),

      
      layers.Dense(num_labels, activation='softmax')
  ])
  return model

adam_opt = tf.keras.optimizers.Adam(learning_rate=0.005,beta_1=0.7,beta_2=0.7)
#sgd_opt = tf.keras.optimizers.SGD(learning_rate=0.05,momentum=0.0)

int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=len(raw_train_ds.class_names))
int_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['categorical_accuracy'])
history = int_model.fit(raw_train_ds, validation_data=raw_val_ds, epochs=10)

print("ConvNet model on int vectorized data:")
print(int_model.summary())

int_loss, int_accuracy = int_model.evaluate(raw_test_ds)
print("Int model accuracy: {:2.2%}".format(int_accuracy))

predictions = int_model.predict_classes(raw_test_ds)

true_labels=[]
for text_bank,label_bank in int_test_ds:
    for label in label_bank:
        true_labels.append(label)

y_true = []
for i in range(len(true_labels)):
    y_true.append(true_labels[i].numpy())

int_model.save("trained_model")

class_names = raw_train_ds.class_names

confusion_chart = tf.math.confusion_matrix(y_true,predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_chart, xticklabels=class_names, yticklabels=class_names, annot=True, fmt='g')
plt.xlabel('Predicted Language')
plt.ylabel('True Language')
plt.show()
