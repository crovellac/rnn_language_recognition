# rnn_language_recognition

This is a project for PH 582-002 Novel Computing Paradigms at The University of Alabama.

I attempt to create a LSTM recurrent neural network capable of reading a string of text and classifying it as English, French, or Lowland Scots. Training and Testing data is taken from public domain works of literature.


##Folders
*test : Test data.
*train : Training data.
*trained_model : The trained and exported neural network which can then be read into another program.
*scraping_text : Contains the program used to format text files into a useable form for Keras.

##Programs
*create_network.py : Builds and trains the network, exports the network, then generates and plots a Confusion matrix on the test data.
*test_network_by_dataset.py : Loads the pretrained network, then generates and plots a Confusion matrix on the test data. Use this if you wish to evaluate the model without retrainig it.
*test_network_manual.py : Loads the pretrained network, then lets the user type a sentence for the network to classify.

These programs are designed for Python3 and require the following libraries:
*Tensorflow and Tensorflow.keras
*numpy
*Matplotlib
*seaborn
