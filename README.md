# Logistic_Regression_from_Scratch_on_a_NLP_task
Sentiment analysis is a process of computationally identifying and categorizing opinions expressed in a piece of text, especially in order to determine whether the writer's attitude towards a particular topic, product, etc. is positive, negative, or neutral.

The main *goal* of this project is to `implement the Logistic Regression from scratch for sentiment analysis on tweets dataset`.

Given a tweet, we will decide if it has a positive sentiment or a negative one. Specifically we will:

* Learn how to extract features for logistic regression given some text
* Implement logistic regression from scratch (implement sigmoid, gradient descent, and cost functions)
* Apply logistic regression on a natural language processing task
* Test using your logistic regression
* Perform error analysis

I achieved the following learning goals:
* Sentiment analysis
* Logistic regression
* Data pre-processing
* Calculating word frequencies
* Feature extraction
* Vocabulary creation
* Supervised learning

## Logistic regression
Logistic regression can be used for sentiment analysis by taking a set of labeled data and training a model to predict the sentiment of new, unlabeled data. The model can be used to classify new data points as positive or negative sentiment.<br>
Logistic regression is a supervised machine learning algorithm that can be used to predict a binary outcome. In binary classification, there are two possible outcomes: *positive (1)* and *negative (0)*. It is a statistical method for predicting the probability of an event occurring. In sentiment analysis, logistic regression can be used to predict the likelihood that a given text document contains a positive or negative sentiment.

## Data pre-processing
Data pre-processing is the process of preparing data for analysis. This includes cleaning data to remove missing or invalid values, transforming data to a format that is more suitable for analysis, and creating new variables from existing data.
The data pre-processing steps we use are:
* Eliminate handles and URLs.
* Tokenize the string into words.
* Remove stop words like "and, is, a, on, etc."
* Stemming- or convert every word to its stem. Like the word "run" is the root of the word family, which includes 'running,' 'ran,''runner,' and 'runs.' You can use porter stemmer to take care of this.
* Convert all your words to lower case.

## Vocabulary creation
After we perform preprocessing, we represent the preprocessed text as a vector by building a **vocabulary** of words and doing **feature extraction** using the vocabulary to convert text into numerical representation. This process is important because it can help to reduce the amount of data that is needed to work with and it can also help to find relationships between the data.<br>
In our case, we use the positive and negative frequencies of the words in the vocabulary to create a frequency dictionary, which maps a word and the class it appeared in (positive or negative) to the number of times that word appeared in its corresponding class.<br>
Hence, we end up with the following feature vector of dimension 3, i.e., [bias, positive_feature, negative_feature]. <br>
The bias term is typically set to 1 in the feature vector to simplify the computation of the dot product between the feature vector and the weight vector (w or θ).

Now, we can use those extracted features to **predict** whether a tweet has a positive or a negative sentiment. 

The logistic regression makes use of **sigmoid function** to map a set of input values to a binary output values(probability between 0 and 1).

*The function is defined as:*
$$sigmoid(x) = \frac{1}{1 + e^{-x}}$$

where x is the input variable and e is the natural logarithm. The output of the sigmoid function is always between 0 and 1.

## Training Logistic Regression

Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).<br>

Steps for training logistic regression model:
1. Collect training data
2. Train the model on the training data
3. Test the model on the test data
4. Evaluate the model's performance<br>

Finally, at this stage, our linear regression classifier is ready, 
and we can determine if it is a good or bad classifier.
<br>

*The first three notebooks are practice lab exercise on particular given topic for better understanding on it. These are recommended to learn for grasp on the concept.*

<a href="https://github.com/yashuv/Logistic_Regression_from_Scratch_on_a_NLP_task/blob/main/Logistic_Regression_from_scratch_for_sentiment_analysis.ipynb">Final Logistic Regression Implementation from Scratch</a>

### Conclusion
I learned to preprocess tweet text and extract features from preprocessed text into numerical vectors, then build a binary classifier for tweets using a logistic regression. I also covered the intuition behind the cost function for logistic regression. <br>
I had the opportunity to apply all of the aforementioned concepts and skills into practice. It was a fantastic learning experience. I appreciate and am grateful for the chance.

Thank you, and happy learning!<br>

---
