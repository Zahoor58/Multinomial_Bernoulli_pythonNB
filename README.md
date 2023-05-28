## Multinomial and Bernoulli Naive Bayes
For understanding Multinomial and Bernoulli Naive Bayes, I Have took a few sentences and classify them in two different classes. Each sentence is  represent one document. In real world examples, every sentence could be a document, such as a mail, or a news article, a book review, a tweet etc. 

The analysis and mathematics involved doesn’t depend on the type of document I used. Therefore I have chosen a set of small sentences to demonstrate the calculation involved and to drive in the concept.
Let us first look at the sentences and their classes. I have kept these sentences in file example_train.csv. Test sentences have been put in the file example_test.csv.
import numpy as np
import pandas as pd
import sklearn 

docs=pd.read_csv('example_train1.csv')
# text in column 1 and classification in column 2
docs
![Uploading image.png…]()

So as you can see there are 5 documents (sentences) , 3 are of "education" class and 2 are of "cinema" class.
# convert label to a numeric variable 
docs['Class']=docs.Class.map({'cinema':0,'education':1})
docs
numpy_array = docs.to_numpy()
X = numpy_array[:,0]
Y = numpy_array[:,1]
Y = Y.astype('int')
print("X")
print(X)
print("Y")
print(Y)
Imagine breaking X in individual words and putting them all in a bag. Then we pick all the unique words from the bag one by one and make a dictionary of unique words. 

This is called **vectorization of words**. I have the class ```CountVectorizer()``` in scikit learn to vectorize the words. Let us first see it in action before explaining it further.

#Create an object of CountVectorizer() class
from sklearn.feature_extraction.text import  CountVectorizer
vec=CountVectorizer()
vec
Here ```vec``` is an object of class ```CountVectorizer()```. This has a method called  ```fit()``` which converts a corpus of documents into a vector of unique words as shown below.
vec.fit(X)
vec.vocabulary_
```Countvectorizer()``` has converted the documents into a set of unique words alphabetically sorted and indexed.


**Stop Words**

We can see a few trivial words such as  'and','is','of', etc. These words don't really make any difference in classyfying a document. These are called 'stop words'. So we would like to get rid of them. 

We can remove them by passing a parameter stop_words='english' while instantiating ```Countvectorizer()``` as follows: 
# Removing the stop_words 
vec=CountVectorizer(stop_words='english')
vec.fit(X)
vec.vocabulary_

Another way of printing the 'vocabulary':
# Printing features names 
print(vec.get_feature_names())
print(len(vec.get_feature_names()))
So our final dictionary is made of 12 words (after discarding the stop words). Now, to do classification, we need to represent all the documents with respect to these words in the form of features. 

Every document will be converted into a *feature vector* representing presence of these words in that document. Let's convert each of our training documents in to a feature vector.
# another way to representing the features
X_transformed =vec.transform(X)
X_transformed
You can see X_tranformed is a 5 x 12 sparse matrix. It has 5 rows for each of our 5 documents and 12 columns each 
for one word of the dictionary which we just created. Let us print X_transformed.
print(X_transformed)
This representation can be understood as follows:

Consider first 4 rows of the output: (0,2), (0,5), (0,7) and (0,11). It says that the first document (index 0) has 
7th , 2nd , 5th and 11th 'word' present in the document, and that they appear only
once in the document- indicated by the right hand column entry. 

Similarly, consider the entry (4,4) (third from bottom). It says that the fifth document has the fifth word present twice. Indeed, the 5th word('good') appears twice in the 5th document. 

In real problems, you often work with large documents and vocabularies, and each document contains only a few words in the vocabulary. So it would be a waste of space to store the vocabulary in a typical dataframe, since most entries would be zero. Also, matrix products, additions etc. are much faster with sparse matrices. That's why we use sparse matrices to store the data.


Let us convert this sparse matrix into a more easily interpretable array:
# converting transformed matrix back to an array 
# Note the high number of zeros 
X=X_transformed.toarray()
X
 To make better sense of the dataset, let us examine the vocabulary and document-term matrix together in a pandas dataframe. The way to convert a matrix into a dataframe is ```pd.DataFrame(matrix, columns=columns)```.

# Converting matrix to Dataframe 
pd.DataFrame(X,columns=vec.get_feature_names())
This table shows how many times a particular word occurs in document. In other words, this is a frequency table of the words.
A corpus of documents can thus be represented by a matrix with one row per document and one column per
token (e.g. word) occurring in the corpus.
We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the "Bag of Words" representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.
#### So, the 4 steps for vectorization are as follows

- Import
- Instantiate
- Fit 
- Transform
Let us summarise all we have done till now:

- ```vect.fit(train)``` learns the vocabulary of the training data
- ```vect.transform(train)``` uses the fitted vocabulary to build a document-term matrix from the training data
- ```vect.transform(test)``` uses the fitted vocabulary to build a document-term matrix from the testing data (and ignores tokens it hasn't seen before)
test_docs = pd.read_csv('example_test.csv') 
#text in column 1, classifier in column 2.
test_docs
# converting label to a numeric variables 
test_docs['Class']=test_docs.Class.map({'cinema':0,'education':1})
test_docs
test_numpy_array=test_docs.to_numpy()
X_test=test_numpy_array[:,0]
Y_test=test_numpy_array[:,1]
print('X_test')
print(X_test)
print('Y_test')
print(Y_test)
X_test_transformed=vec.transform(X_test)
X_test_transformed
X_test=X_test_transformed.toarray()
X_test
## Multinomial Naive Bayes
# Building a multinomial NB model
from sklearn.naive_bayes import MultinomialNB
# instantiate NB class
mnb=MultinomialNB()
# fit the model in training data
mnb.fit(X,Y)
#Predicting probabilities of data
mnb.predict_proba(X_test)
proba=mnb.predict_proba(X_test)
print("probability of test document belonging to class CINEMA" , proba[:,0])
print("probability of test document belonging to class EDUCATION" , proba[:,1])
pd.DataFrame(proba, columns=['Cinema','Education'])
## Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB

# instantiating bernoulli NB class
bnb=BernoulliNB()

# fitting the model
bnb.fit(X,Y)

# predicting probability of test data
bnb.predict_proba(X_test)
proba_bnb=bnb.predict_proba(X_test)
pd.DataFrame(proba_bnb, columns=['Cinema','Education'])






