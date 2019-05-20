## DeepBug
Implementation of DeepBug with Keras framework.

## Setup
 1. Python version of 3.6.x or later.
 2. Install required packages.
        `pip install -U nltk gensim tensorflow keras scikit-learn`
 3. Download the full project from github
        `git clone https://github.com/lofus/deepbug`
 4. Unzip ./data/google_chromium/all_data_20.npy.zip
 5. Now you can run train_and_evaluate_cnn_model.py to train and and evaluate the model

## Project files
- `dataset.py` load dataset, reading and slicing methods for chronological cross validation.
- `train_and_evaluate_cnn_model.py` run this script to train and evaluate the model
- `deepbug_cnn_model.py` model implementation with Keras.

Here is a snapshot for the model summary, as printed from log
model summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 50, 200, 1)        0         
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 56, 206, 1)        0         
_________________________________________________________________
bn0 (BatchNormalization)     (None, 56, 206, 1)        824       
_________________________________________________________________
conv0 (Conv2D)               (None, 52, 202, 16)       416       
_________________________________________________________________
activation_1 (Activation)    (None, 52, 202, 16)       0         
_________________________________________________________________
max_pool0 (MaxPooling2D)     (None, 26, 101, 16)       0         
_________________________________________________________________
conv1 (Conv2D)               (None, 24, 99, 32)        4640      
_________________________________________________________________
activation_2 (Activation)    (None, 24, 99, 32)        0         
_________________________________________________________________
max_pool1 (MaxPooling2D)     (None, 12, 49, 32)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 12, 49, 32)        0         
_________________________________________________________________
conv2 (Conv2D)               (None, 10, 47, 64)        18496     
_________________________________________________________________
activation_3 (Activation)    (None, 10, 47, 64)        0         
_________________________________________________________________
max_pool2 (MaxPooling2D)     (None, 5, 23, 64)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 23, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 7360)              0         
_________________________________________________________________
fc0 (Dense)                  (None, 2000)              14722000  
_________________________________________________________________
softmax (Dense)              (None, 781)               1562781   
=================================================================
Total params: 16,309,157
Trainable params: 16,308,745
Non-trainable params: 412


## Datasets
The link for the datasets:

- [Google Chromium](https://drive.google.com/file/d/0Bz07ySZGa87tdENrZjAxelBPdFE/view) - 383,104 bug reports

A sample bug report from datasets is given below:

#### Google Chromium:
```json
{
		"id" : 1,
		"issue_id" : 2,
		"issue_title" : "Testing if chromium id works",
		"reported_time" : "2008-08-30 16:00:21",
		"owner" : "jackm@chromium.org",
		"description" : "\nWhat steps will reproduce the problem?\n1.\n2.\n3.\n\r\nWhat is the expected output? What do you see instead?\n\r\n\r\nPlease use labels and text to provide additional information.\n \n ",
		"status" : "Invalid",
		"type" : "Bug"
}
```
