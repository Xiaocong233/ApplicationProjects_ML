## Description
- contains a series of working/finished Machine Learning-applied problems and projects

## NimAI
- AI learning to maximize its value in the game of Nim through Reinforcement Learning
  - trained with Q-Learning with 10000 rounds of playing
  - absolutely dominates the human player

## OnlineShopping
- AI predicting whether a customer will complete to purchase a product based on features collected
  - ML model trained with K-nearest Neighbors (k = 1)
  
## Query Bot
- AI that reads through a corpus of text files provided on relevant topics and automatically responds to the player's query according to the best matched the sentence in the best matched text file
  - trained using nltk with files ranked based on tf-idf values and sentences ranked based on idf values and query term density
  - example:
    '''
      $ python questions.py corpus
      Query: What are the types of supervised learning?
      Types of supervised learning algorithms include Active learning , classification and regression.
    '''
      
    '''
      $ python questions.py corpus
      Query: When was Python 3.0 released?
      Python 3.0, released in 2008, was a major revision of the language that is not completely backward-compatible, and much Python 2 code does not run unmodified on Python 3.
    '''
