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

def space_out_string(string):
    outstring = ""
    for character in string:
        if (character == " "):
            outstring += "_"
        else:
            outstring += character
        outstring += " "
    return outstring

model = tf.keras.models.load_model("trained_model")

raw_train_ds = preprocessing.text_dataset_from_directory(
    "train",
   )

class_names = raw_train_ds.class_names


raw_test_ds = preprocessing.text_dataset_from_directory("test", batch_size=batch_size_choice)

predictions = model.predict_classes(raw_test_ds)

true_labels=[]
for text_bank,label_bank in raw_test_ds:
    for label in label_bank:
        true_labels.append(label)


y_true = []
for i in range(len(true_labels)):
    y_true.append(true_labels[i].numpy())


confusion_chart = tf.math.confusion_matrix(y_true,predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_chart, xticklabels=class_names, yticklabels=class_names, annot=True, fmt='g')
plt.xlabel('Predicted Language')
plt.ylabel('True Language')
plt.show()
