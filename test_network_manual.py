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


while(True):
    print("---------------------------------")
    print("Type a sentence and press 'enter' to see what the model guesses.")
    original_in_text = input("")
    in_text = space_out_string(original_in_text)
    input_vec = [in_text]

    prediction = model.predict_classes(input_vec)
    print("---------------------------------")
    print("Input Text: "+original_in_text)
    print("Predicted Language: "+class_names[int(prediction[0])])
