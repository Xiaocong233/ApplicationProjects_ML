import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    fileDict = dict()

    for _, _, filenames in os.walk(directory):
        for filename in filenames:
            text = open(os.path.join(directory, filename), 'r', encoding = 'UTF-8').read()
            fileDict[filename] = text

    return fileDict
        

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # tokenize all words in the documents
    words = list(nltk.word_tokenize(document.lower()))

    # filtering out punctuation and stopwords
    punc = string.punctuation
    stopwords = nltk.corpus.stopwords.words('english')
    for word in words:
        if word in punc or word in stopwords:
            words.remove(word)

    return words

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # empty list for idfs
    idfs = dict()

    # get the total numbers of documents
    numDocuments = len(documents.keys())

    # loop through each word (ignore repetition) in each doc, finding and appending its idf value
    for doc in documents.values():
        for word in doc:
            if word not in idfs.keys():
                numAppearances = 0
                for doc2 in documents.values():
                    if word in doc2:
                        numAppearances += 1
                idfs[word] = math.log(numDocuments / numAppearances)

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # initialize a dictionary for score
    score = dict()

    # loop through each file and word in the query and update the score of each file
    for file in files.keys():
        # initialize all score to be 0
        score[file] = 0
        for word in query:
            tf = files[file].count(word)
            score[file] += tf * idfs[word]

    # add to and sort the file rank based on each file's score
    file_rank = list(sorted(score, key = score.get, reverse = True))

    # return the first n elements
    return file_rank[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # creat a dicts storing sentences score and their query term density as lists
    scores = dict()

    # loop through the sentences and find/rank them by their idf values
    for sentence in sentences:
        scores[sentence] = [0, 0]
        for word in query:
            if word in sentences[sentence]:
                scores[sentence][0] += idfs[word]
                scores[sentence][1] += 1 / len(sentences[sentence])

    # add to and sort the sentence rank based on each file's score and dens_score
    sentence_rank = list(sorted(scores, key = lambda sentence: (scores[sentence][0], scores[sentence][1]), reverse = True))
        
    # return the first n elements
    return sentence_rank[:n]

if __name__ == "__main__":
    main()
