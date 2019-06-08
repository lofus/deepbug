## DeepBug
We propose a deep neural network approach named as DeepBug for software bugs triage.
DeepBug takes the bug report title, description and logs as input features, and pre
predict that to which software component the bug belongs to. Deepbug is designed as
a hybrid architecture with a  CNN and a RNN running in parallel. The CNN path keeps 
learning from the bug log file in a way mimicking manual log analysis by layers,and
the RNN path works on learning the text sequences from bug title and description.The
stream of the two paths are aggregated as the final output as a classifier.

## Setup
 1. Python version of 3.6.x or later.
 2. Install required packages.
        `pip install -U nltk gensim tensorflow keras scikit-learn`
 3. Download the full project from github
        `git clone https://github.com/lofus/deepbug`
 4. build the dataset, this will depend on the data source for bugreports database
 5. run data_preprocess.py to get dataset cleanse and ready.
 6. run train_and_evaluate_deepbug.py to train and validate DeepBug.

## Project files
- `build_dataset.py` load dataset, reading and slicing methods for cross validation.
- `deepbug_models.py` models implementation with Keras.
- `train_and_evaluate_deepbug.py` run this script to train and evaluate the model

## Datasets
Public dataset:
- [Google Chromium](https://drive.google.com/file/d/0Bz07ySZGa87tdENrZjAxelBPdFE/view)

##
We train and validate DeepBug with the industry dataset and our internal data for bug
reports from JIRA system with a total of 200K samples, where we explicitly set the fe
ature ‘owner’ to be the software component name against which the bug was resolved.

The data features and label are listed as following.
● Title -- This is a summary for the bug, usually it is less than 30 words.
● Description -- A description of what are the scenarios and behaviors from user
experience and or testing observation. This could be more than 300 words.
● Logs -- This can be an attachment such as log files, which is a snapshot to the
software stack, a printed from each layer of the Android software stack.
● Component -- This is the actual data label . For a closed ticket, this is the 
component name where the bug was finally fixed. For a new ticket, this means the
component name that DeepBug works to predict for.

