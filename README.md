# NLP-Projects
## Text Classification of twitter data
Projects that involve Natural Processing Language applications like Classification, Sentiment Analysis, Pre-processing etc

This is a simple project based on NLP- Text Classification of twitter data
It shows the various steps which are involved during text classification and how a Machine Learning Model can be trained and tested.

Python Libraires used: Pandas, re (for Regular Expressoins) , nltk (Natural Language Toolkit), Scikit-Learn

Label-0 : Non Racist Tweets

Lable-1 : Racist Tweets

Project Workflow:

1) Import all the necessary libraries

2) Load the dataset (csv file) using pandas

3) Perform Data Cleaning
        
        3.1) Only keep alphabets of the tweet text
        3.2) Remove the unicode characters as they are useless
        3.3) Convert the text into lower case for consistency
        
4) Feature Engineering
   Feature engineering is the science (and art) of extracting more information from existing data. 
   We are not adding any new data here, but we are actually making the data we already have more useful.
   The machine learning model does not understand text directly, so we create numerical features that reperesant the underlying text.
   
        4.1) Generate the word frequency
        4.2) Check whether a negation term is present in the text
        4.3) Check whether one of the 100 rare words is present in the text
        4.4) Check whether prompt words are present
        
5) Creation of numerical features in the dataset like---> word_count,any_neg,any_rare,char_count,is_question

6) Splitting the dataset into Train-Test split

7) Train an ML model for Text Classification- used NaiveBayes Classifier from sklearn

8) Evaluate the ML model- using various metrics lke classification_report , accuracy score

Result: Accuracy Score was found to be 60%

**) The various codes in comments can also be executed to get a better understanding of the workflow in this project
