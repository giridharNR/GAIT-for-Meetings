# GAIT: Generating Action Items using Transformers

This repository provides the code for automatically generating a document of action items from online meeting transcripts.

It consists of 3 parts,

## Topic Segmentation
This involves running the meeting transcript through a transformer model (CATS) to segment it into the different topics discussed. The CATS model takes text as input and provides a probability of each sentence denoting the end of a topic. We then divide the meeting into topics by applying an adaptive threshold on the probabilities. Please look into the `CATS\` folder for more details.

## Action Item Classification
This section involves classification of each sentence to denote whether or not it denotes an action item. Several models are used for this purpose. The transformer based BERT model gives the best performance.

## Summarization
This section involves summarizing the action items to a more compact and readable form. We use BART model, which is the state-of-the-art for Summarization tasks.

### Authors: 
Soham Hans (sohamhan@usc.edu), Giridhar NR (narasapu@usc.edu), Richard (myloth@usc.edu), Manasa Rajesh (mrajesh@usc.edu), Xinyu Shen (shenxiny@usc.edu)
