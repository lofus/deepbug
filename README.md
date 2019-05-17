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
X_train.shape (18885, 50, 200, 1)
y_train.shape (18885, 865)
X_test.shape (1000, 50, 200, 1)
y_test.shape (1000, 865)
classes: 865
deepbug_model_cnn start..
input_shape: (50, 200, 1)
num_output: 865
model summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 50, 200, 1)        0         
_________________________________________________________________
zero_padding2d_2 (ZeroPaddin (None, 56, 206, 1)        0         
_________________________________________________________________
conv0 (Conv2D)               (None, 25, 100, 16)       800       
_________________________________________________________________
bn0 (BatchNormalization)     (None, 25, 100, 16)       400       
_________________________________________________________________
activation_2 (Activation)    (None, 25, 100, 16)       0         
_________________________________________________________________
max_pool (MaxPooling2D)      (None, 12, 25, 16)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 4800)              0         
_________________________________________________________________
fc (Dense)                   (None, 865)               4152865   
=================================================================
Total params: 4,154,065
Trainable params: 4,153,865
Non-trainable params: 200
_________________________________________________________________




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
