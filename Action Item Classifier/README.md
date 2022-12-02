This folder contains code for the action item classifier. `training\` folder contains code for training the models and the dataset used for training models. `models\` folder contains the trained model for doing the classification.


## Setup

In addition to commonly used NLP libraries (pandas, torch, sklearn). We used `transformers` library in our project for the BERT model. Another library used for expanding contractions is `contractions`.

Install the libraries using the following command:
* `pip install transformers`
* `pip install contractions`

## Identifying the Action Sentences

To identify the action sentences from the segmented results from CATS, run the following command.
* `python action_classifier.py [OPTIONS] [INPUT] [OUTPUT]`

INPUT: Path to the input file containing the segmented results from CATS. Each line represents a topic. An example input file is `segmented-1.txt`
OUTPUT: Path to the output file containing the action sentences. Each line contains the action sentences from a topic.
OPTIONS:
* Bag of Words + Artificial Neural Network: `-ann`
* Random Forest: `-rf`
* BERT: `-tr`


