# Naive Bayes Algorithms

This repository contains implementations of the Multinomial and Bernoulli Naive Bayes algorithms for text classification tasks.

## Introduction
The Naive Bayes algorithm is a classification algorithm based on Bayes' theorem. It assumes independence between features, making it efficient and suitable for various applications such as text classification and spam filtering.

## Multinomial Naive Bayes
The Multinomial Naive Bayes algorithm is specifically designed for discrete features, commonly used in text classification. It works by counting the occurrences of words in documents and calculating probabilities based on these counts. The algorithm is trained on a labeled dataset, and then it can classify new documents by estimating the probabilities of different classes given the features.

### Code Example
Here's an example of using the Multinomial Naive Bayes algorithm in Python:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load training data
X_train = [...]  # List of training documents
y_train = [...]  # List of corresponding labels

# Create feature vectors
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Classify new documents
X_test = [...]  # List of new documents
X_test_vectorized = vectorizer.transform(X_test)
predictions = clf.predict(X_test_vectorized)
```

### Bernoulli Naive Bayes
The Bernoulli Naive Bayes algorithm is a variation of Naive Bayes suitable for binary features. It considers only the presence or absence of a feature, making it useful for tasks where the occurrence of a feature is important. For example, it can be used for sentiment analysis or spam detection.

### Code Example
Here's an example of using the Bernoulli Naive Bayes algorithm in Python:
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# Load training data
X_train = [...]  # List of training documents
y_train = [...]  # List of corresponding labels

# Create binary feature vectors
vectorizer = CountVectorizer(binary=True)
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the classifier
clf = BernoulliNB()
clf.fit(X_train_vectorized, y_train)

# Classify new documents
X_test = [...]  # List of new documents
X_test_vectorized = vectorizer.transform(X_test)
predictions = clf.predict(X_test_vectorized)
```

### Advantages and Limitations
The Naive Bayes algorithms have several advantages, including simplicity, fast training and prediction times, and the ability to handle high-dimensional data efficiently. However, they assume independence between features, which might not hold in all cases. Additionally, they can encounter issues with zero probabilities, requiring the use of techniques like Laplace smoothing.

### Conclusion
The Multinomial and Bernoulli Naive Bayes algorithms are powerful tools for text classification and other tasks involving discrete or binary features. By understanding and implementing these algorithms, you can perform efficient and accurate classification on your datasets.
