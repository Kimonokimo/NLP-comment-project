# Toxic Comment Classification
This project focuses on establishing a classification model to identify and classify the toxic comments on Wikipedia. 
The project was done by Qimo Li, Chen He and Kun Qiu for the course "Introduction to Natural Language Processing in Python" at 
Brandeis University. 

This project is based on the Kaggle Toxic Comment Classification Challenge. The dataset is shown as `train.csv`. For more information about the challenge: 
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
 
This project contains the following files:
* The `dataclean.py` script is used to remove punctuation, numbers, and abbreviation.
* The `tokenizer.py` script creates a Tokenizer class that tokenizes raw text and sentences.
* The `pos_tagger.py` script creates a pos tagger using the Random Forest model.
* The `lemmatizer.py` script creates a WordNet-based lemmatizer.
* The `vector.py` script is used to vectorize the comments based on the given vocabulary, 
calculate term frequency(tf) value, and return ndarrays.
* The `vocabulary.py` script creates a vocabulary from the dataset with stop words removed. The `Vocabulary` class
contains method for computing the inverse document frequency (idf) value for each word in the vocabulary.
* The `visualization.ipynb` script creates several basic data exploratory visualizations and creates interactive HTML
visualization using `scattertext` library.
* The `visualization` folder contains the HTML visualization created by `visualization.ipynb`. The resulting screenshots of the HTML visualization are in the `visualization_results_screenshots` folder.

The main method can be executed in the following file:
* The `main.py` script is the core of the project. It loads the data set, cleans and parses the comments, vectorizes 
the text, encodes tf-idf values, splits the data for cross validation and builds an ensemble classification model 
consists of six logistic regression models. The `main.py` script also contains methods for splitting training and 
testing data, mapping tag sets, and evaluating the performance.

This project requires the following libraries:
* re
* nltk
* pandas
* numpy
* os
* joblib
* sklearn
* scattertext
* IPython
* spacy

For motivations, implementation details and results, please check out the project `report.pdf` file. 

### How to run our code: 
`$ python main.py`
